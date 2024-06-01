import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from dataclasses import dataclass
import logging
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DiffusionPipeline
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from diffusers import UNet2DConditionModel
from accelerate import Accelerator
from tqdm.auto import tqdm

# Initialize logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")

# Define the device (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_info(message):
    logging.info(message)
    print(message)

def log_error(message):
    logging.error(message)
    print(f"Error: {message}")

# Define a function to calculate PSNR score
def calculate_psnr(real_images, generated_images):
    total_psnr = 0.0
    for real_img, gen_img in zip(real_images, gen_img):
        total_psnr += psnr(real_img, gen_img)
    avg_psnr = total_psnr / len(real_images)
    return avg_psnr

# Define your dataset and data loader
@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 4  # 16
    eval_batch_size = 4  # 16 how many images to sample during evaluation
    num_epochs = 1  # 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1  # 10
    save_model_epochs = 1  # 30
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "data/datasets"  # the model name locally and on the HF Hub
    seed = 0

config = TrainingConfig()

def open_image_if_jpg_or_jpeg(image_path):
    # Check if the file path ends with .jpg or .jpeg (case-insensitive)
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        try:
            # Try to open the image
            image = Image.open(image_path)
            return image
        except Exception as e:
            # Print an error message if the image cannot be opened
            print(f"Error opening image {image_path}: {e}")
            return None
    else:
        # Skip the file if it does not have a .jpg or .jpeg extension
        print(f"Skipped file: {image_path} (not a .jpg or .jpeg)")
        return None

# Define datasets for training without glasses and with glasses
class CustomDataset(Dataset):
    def __init__(self, dataset_path, landmarks_file, mask_dir, with_glasses=True):
        assert os.path.exists(dataset_path), f"dataset path: {dataset_path} doesn't exist, please pass correct path"
        self.dataset = [os.path.join(dataset_path, img_path) for img_path in os.listdir(dataset_path)]
        self.landmarks = self.load_landmarks(landmarks_file)
        self.mask_dir = mask_dir
        self.with_glasses = with_glasses

    def load_landmarks(self, landmarks_file):
        landmarks = {}
        with open(landmarks_file, 'r') as f:
            # Skip the first two lines which contain header information
            for _ in range(2):
                next(f)
            for line in f:
                parts = line.strip().split()
                filename = parts[0]
                left_eye_x, left_eye_y, right_eye_x, right_eye_y = map(int, parts[1:5])
                landmarks[filename] = [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
        return landmarks

    def create_mask(self, landmarks, image_size):
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        # Get left and right eye coordinates
        left_eye_x, left_eye_y, right_eye_x, right_eye_y = landmarks
        # Calculate mask dimensions around the eyes
        expand_factor = 0.25  # 25% expansion
        width = int(abs(right_eye_x - left_eye_x) * (1 + expand_factor))
        height = int(abs(right_eye_y - left_eye_y) * (1 + expand_factor))
        x = int(left_eye_x - (width - (right_eye_x - left_eye_x)) / 2)
        y = int(left_eye_y - (height - (right_eye_y - left_eye_y)) / 2)

        # Ensure width and height of axes are non-negative
        assert width >= 0 and height >= 0, "Width and height of axes must be non-negative."

        # Ensure thickness is within valid range
        max_thickness = 10  # Adjust as needed
        thickness = 2  # Set the thickness here
        assert 0 <= thickness <= max_thickness, "Thickness must be between 0 and max_thickness."

        # Ensure shift is within valid range
        XY_SHIFT = 16  # Value from OpenCV, adjust if needed
        shift = 0
        assert 0 <= shift <= XY_SHIFT, "Shift must be between 0 and XY_SHIFT."

        # Draw filled ellipse around the eyes
        # cv2.ellipse(mask, ((x + right_eye_x) // 2, (y + right_eye_y) // 2),
        #             ((width + right_eye_x - left_eye_x) // 2, (height + right_eye_y - left_eye_y) // 2),
        #             # 0, 0, 360, (255, 255, 255), thickness)
        #             0, 0, 360, (255, 255, 255), -1)
        center = (int((x + right_eye_x) // 2), int((y + right_eye_y) // 2))
        axes = (int((width + right_eye_x - left_eye_x) // 2), int((height + right_eye_y - left_eye_y) // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255), -1)

        return mask

    def load_mask(self, image_name, landmarks, image_size):
        mask_path = os.path.join(self.mask_dir, image_name)
        print(f"mask path: {mask_path}")
        if not os.path.exists(mask_path):
            # Mask file not found, create mask automatically
            mask = self.create_mask(landmarks, image_size)
            # mask = np.expand_dims(mask, 0)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # mask = np.expand_dims(mask, 0)
            if mask is None:
                # print("Failed to load mask file!")
                mask = self.create_mask(landmarks, image_size)
            else:
                mask = cv2.resize(mask, (image_size, image_size))
        mask = np.expand_dims(mask, 0)
        return mask

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx]
        # image = Image.open(image_path)
        image = open_image_if_jpg_or_jpeg(image_path)
        if image is None:
            # Skip this index if the image couldn't be opened
            print(f"Skipped index {idx} because image could not be opened")
            return None  # Or raise an exception, depending on how you want to handle this

        image = self.preprocess_image(image)

        # Print the shape of the input image
        print("Image shape:", image.shape)

        image_name = os.path.basename(image_path)
        landmarks = self.landmarks.get(image_name)  # Use .get() to handle missing landmarks gracefully
        if landmarks is None:
            # If landmarks are missing, handle it gracefully
            log_error(f"No landmarks found for image: {image_name}")
            return None

        landmarks = torch.tensor(landmarks)
        # mask = self.load_mask(image_name, landmarks, config.image_size) if self.with_glasses else torch.ones(1, *image.shape[1:])
        mask = self.load_mask(image_name, landmarks, config.image_size) if self.with_glasses else torch.ones(1, config.image_size, config.image_size)

        return {'image': image, 'landmarks': landmarks, 'mask': mask, 'with_glasses': self.with_glasses}

class CustomUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, in_channels=7, **kwargs):
    # def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Override the first convolutional layer to accept the correct number of input channels
        self.conv_in = torch.nn.Conv2d(in_channels, self.conv_in.out_channels, kernel_size=self.conv_in.kernel_size, padding=self.conv_in.padding)

    def forward(self, sample, landmarks, timesteps, return_dict=True):
        # Normalize landmarks by image size
        landmarks = landmarks.float() / sample.size(2)
        landmarks = landmarks.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, sample.size(2), sample.size(3))
        
        # Concatenate landmarks with the input sample along the channel dimension
        sample = torch.cat([sample, landmarks], dim=1)
        encoder_hidden_states = landmarks

        # Forward pass through the UNet model
        output = super().forward(sample, timesteps, encoder_hidden_states, return_dict=return_dict)
        return output

def evaluate(config, epoch, pipeline):
    try:
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed),
        ).images

        # Save each inpainted image separately
        test_dir = os.path.join(config.output_dir, "inpainted_results")
        os.makedirs(test_dir, exist_ok=True)
        for i, img in enumerate(images):
            img.save(f"{test_dir}/{epoch:04d}_{i:03d}.png")
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, is_glasses):
    try:
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=config.output_dir,
        )
        if accelerator.is_main_process:
            accelerator.init_trackers("train_example")

        global_step = 0
        for epoch in range(config.num_epochs):
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
                if batch is None:
                    continue
                clean_images = batch['image']
                landmarks = batch['landmarks']
                mask = batch['mask']
                
                print(f"clean_images shape: {clean_images.shape}, landmarks shape: {landmarks.shape}")

                clean_images = clean_images.to(accelerator.device)
                landmarks = landmarks.to(accelerator.device)
                mask = mask.to(accelerator.device)

                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                print(f"noisy_images shape: {noisy_images.shape}")

                # Predict the noise residual
                noise_pred = model(noisy_images, landmarks, timesteps).sample
                
                print(f"noise_pred shape: {noise_pred.shape}")

                loss = F.mse_loss(noise_pred, noise)

                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % 10 == 0:
                    log_info(f"Epoch [{epoch}/{config.num_epochs}], Step [{step}/{len(train_dataloader)}], Loss: {loss.item()}")

            # Save model and evaluate every few epochs
            if (epoch + 1) % config.save_model_epochs == 0:
                pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
                pipeline.save_pretrained(config.output_dir)
                evaluate(config, epoch, pipeline)
    except Exception as e:
        log_error(f"Training loop error: {e}")
        raise

def main():
    dataset_without_glasses = CustomDataset("data/datasets/without_glasses", "data/datasets/list_landmarks_align_celeba.txt", "data/datasets/glasses_masks", with_glasses=False)
    dataset_with_glasses = CustomDataset("data/datasets/with_glasses", "data/datasets/list_landmarks_align_celeba.txt", "data/datasets/glasses_masks", with_glasses=True)

    # combined_dataset = dataset_without_glasses + dataset_with_glasses
    combined_dataset = [item for item in dataset_without_glasses if item is not None] + [item for item in dataset_with_glasses if item is not None]
    train_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=config.train_batch_size, shuffle=True)

    # model = CustomUNet2DConditionModel.from_pretrained("stable-diffusion-v1-4", subfolder="unet",  in_channels=9, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    model = CustomUNet2DConditionModel.from_pretrained("stable-diffusion-v1-4", subfolder="unet",  in_channels=7, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    noise_scheduler = DDPMScheduler.from_config("stable-diffusion-v1-4")
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=len(train_dataloader) * config.num_epochs)

    # Train the model
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, True)

if __name__ == "__main__":
    main()
