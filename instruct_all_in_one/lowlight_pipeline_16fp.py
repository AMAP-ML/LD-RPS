import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import argparse
import numpy as np
import requests
import time
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from utils import *
#ganlr=4e-5, deg_lr=5e-4
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline, 
    StableDiffusionDepth2ImgPipeline,
    LDMTextToImagePipeline
    )
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler
from diffusers import LMSDiscreteScheduler

def create_args():
    parser = argparse.ArgumentParser()
    #add_dict_to_argparser(parser, defaults)
    parser.add_argument("--inference_num", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_bn", action='store_true')
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--degradation_para", type=list, default=[0.02, 0.3, 0.8])
    parser.add_argument("--guidance_scale", type=int, default=80000)
    parser.add_argument("--iter", type=int, default=2)
    parser.add_argument("--result_dir", type=str, default='./results/enhacement')
    parser.add_argument("--sample_dir", type=str, default='./data/LOLv1/input')
    parser.add_argument("--text_dir", type=str, default='./data/LOLv1/prompt')
    return parser.parse_args()

def get_latent_space(img, pipe):
    '''
    image.shape:([b, 3, 512, 512]~(-1,1))
    out: latent_space.shape [b, 4, 64, 64]
    '''
    latents = 0.18215 * pipe.vae.encode(img.to(torch.float16)).latent_dist.mean

    return latents

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"set the random seed to {str(seed)}")

def check_value(x):
    return x if x < 1 else 1

def general_cond_fn(x, t, lq_latent=None, lq_image=None):
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        x_in_hq = pipe.vae.decode(x_in / 0.18215, return_dict=False, generator=generator)[0]
        x_in_hq = ((x_in_hq + 1) / 2).to(torch.float32)
        x_tg_lq = ((lq_image + 1) / 2).to(torch.float32)
        
        #添加latent预处理方式
        x_in_lq = degeneration_model(x_in_hq, x_tg_lq, t)
        x_lq = (x_in_lq*2 - 1).to(torch.float32)
        learned_lq = get_latent_space(x_lq, pipe)
        """ with torch.no_grad():
            image = pipe.decode_latents(learned_lq[0:1].detach())
            image = (image[0] * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save('learned_lq.png')
        with torch.no_grad():
            image = pipe.decode_latents(learned_lq[1:].detach())
            image = (image[0] * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save('learned_lq2.png') """

        # First Stage (t ~ [999, 600])
        mse = mse_loss(x_in_lq, x_tg_lq.detach()) * 0.0 if t >= 700 else 0
        warmup = check_value((700. - t)/100.)
        mse = mse_loss(x_in_lq[:args.batch_size],x_tg_lq[:args.batch_size].detach()) * 1.0 if t < 700 else mse
        pse = perceptual_loss(x_in_lq[:args.batch_size, :3],
                                x_tg_lq[:args.batch_size, :3].detach()) * 2e-3 if t < 700 else 0
        adv = adversarial_loss((x_in_hq - x_tg_lq.detach())[:args.batch_size],
                                (x_in_hq - x_in_lq).detach()) * 5e-5 if t < 700 else 0
        exp = exp_loss(x_in_hq[:args.batch_size]) * 1e-2 if t < 100 else 0
        col = col_loss(x_in_hq[:args.batch_size]) * 1e-2 if t < 100 else 0
        
        loss = args.guidance_scale * (mse + pse + adv + exp + col)
        #print(type(loss))

        guidance_loss_dict = {"mse": mse, "pse": pse, "adv": adv, "exp": exp, "col": col}

        gui_log, deg_log = "", ""
        for k, v in guidance_loss_dict.items(): gui_log += f"{k}={v:.5f}, "
        for k, v in degeneration_model.loss_dict.items(): deg_log += f"{k}={v:.5f}, "
        print(f"step={t:d}, dege({deg_log[:-2]}), guid(scale={args.guidance_scale:d}, {gui_log[:-2]})")
        gradient = torch.autograd.grad(-loss, x_in)[0]
        print(gradient.mean().item())
        
        return gradient

def condition_mean(cond_fn, latents, pred_xstart, t):
    """
    Compute the mean for the previous step, given a function cond_fn that
    computes the gradient of a conditional log probability with respect to
    x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
    condition on y.

    This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
    """
    # gradient = cond_fn(pred_xstart, self._scale_timesteps(t), **model_kwargs)
    #optimizer = th.optim.Adam([latents], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    new_mean = latents.float()
    for i in range(1):
        gradient = cond_fn(pred_xstart, t)
        new_mean = (latents.float() +  gradient.float())
        """ new_mean.requires_grad_(True)
        optimizer.zero_grad()
        new_mean.backward(th.ones_like(new_mean))
        optimizer.step() """
    return new_mean[0:1], new_mean[1:2]

def training(timesteps, pipe, latents, cond_fn, text_embeddings):
    latents1, latents2 = latents[0:1], latents[1:2]

    for i, t in enumerate(timesteps):
        print(t)

        # Expand the latents if we are doing classifier free guidance
        latent_model_input1 = torch.cat([latents1] * 2)
        latent_model_input2 = torch.cat([latents2] * 2)

        # Apply any scaling required by the scheduler
        latent_model_input1 = pipe.scheduler.scale_model_input(latent_model_input1, t)
        latent_model_input2 = pipe.scheduler.scale_model_input(latent_model_input2, t)

        # Predict the noise residual with the UNet
        with torch.no_grad():
            noise_pred1 = pipe.unet(latent_model_input1, t, encoder_hidden_states=text_embeddings).sample
            noise_pred2 = pipe.unet(latent_model_input2, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        noise_pred_uncond1, noise_pred_text1 = noise_pred1.chunk(2)
        noise_pred_uncond2, noise_pred_text2 = noise_pred2.chunk(2)
        noise_pred1 = noise_pred_uncond1 + guidance_scale * (noise_pred_text1 - noise_pred_uncond1)
        noise_pred2 = noise_pred_uncond2 + guidance_scale * (noise_pred_text2 - noise_pred_uncond2)

        # Compute the previous noisy sample x_t -> x_t-1
        latents1 = pipe.scheduler.step(noise_pred1, t, latents1).prev_sample
        latents2 = pipe.scheduler.step(noise_pred2, t, latents2).prev_sample
        
        pred_xstart1 = pipe.scheduler.step(noise_pred1, t, latents1).pred_original_sample
        pred_xstart2 = pipe.scheduler.step(noise_pred2, t, latents2).pred_original_sample

        #latents = condition_mean(cond_fn, latents, pred_xstart, t)
        pred_xstart = torch.cat([pred_xstart1, pred_xstart2])
        latents = torch.cat([latents1, latents2])
        latents1, latents2 = condition_mean(cond_fn, latents, pred_xstart, t)
        latents1 = latents1.to(torch.float16)
        latents2 = latents2.to(torch.float16)
        """ with torch.no_grad():
            image = pipe.decode_latents(latents1.detach())
            image = (image[0] * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save('latents1.png') """
        """ with torch.no_grad():
            image = pipe.decode_latents(latents2.detach())
            image = (image[0] * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save('latents2.png') """

        #img = pipe.vae.decode(pred_xstart / 0.18215, return_dict=False, generator=generator)[0]
        """ with torch.no_grad():
            image = pipe.decode_latents(pred_xstart1.detach())
            image = (image[0] * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save('pred_x0_lowlight.png')
        with torch.no_grad():
            image = pipe.decode_latents(pred_xstart2.detach())
            image = (image[0] * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save('pred_x0_lowlight2.png') """

    return torch.cat([latents1, latents2])

args = create_args()
device = "cuda" if torch.cuda.is_available else 'cpu'
tv_loss = TVLoss().to(device)
mse_loss = MSELoss().to(device)
perceptual_loss = VGGLoss().to(device)
adversarial_loss = GANLoss().to(device)

if args.use_bn:
    degeneration_model = GenerativeDegradation_res(input_para=args.degradation_para).to(device)
else:
    degeneration_model = GenerativeDegradation(input_para=args.degradation_para).to(device)

exp_loss = L_exp(32, 0.5).to(device)
col_loss = L_color().to(device)

model_id = "/mnt/workspace/common/models/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
# Replace the scheduler
config = pipe.scheduler.config
config['thresholding'] = True
config['sample_max_value'] = 100.
config['dynamic_thresholding_ratio'] = 0.995
pipe.scheduler = DDIMScheduler.from_config(config)
img_pipe.scheduler = DDIMScheduler.from_config(config)

num_inference_steps = 450
img_num_inference_steps = 450
guidance_scale = 8
strength = 0.5

os.makedirs(args.result_dir, exist_ok=True)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

# 遍历图像
for file_name in os.listdir(args.sample_dir):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    init_seed(args.seed)
    print(file_name)
    lq_path = os.path.join(args.sample_dir, file_name)
    prompt_path = os.path.join(args.text_dir, file_name.split('.')[0]+'.txt')
    with open(prompt_path, 'r') as f:
        prompt = f.read().strip() 
        print(prompt)
    # a diy loop
    #prompt = ''
    negative_prompt = ''

    lq_image = load_img(lq_path).to(device)
    lq_image = lq_image.repeat(2, 1, 1, 1)
    #lq_latent = get_latent_space(lq_image, pipe)
    lq_latent = lq_image

    adversarial_loss.init_disc_net()
    degeneration_model.init_gene_net()
    cond_fn = lambda x, t: general_cond_fn(x, t, lq_latent, lq_image)

    # Encode the prompt
    text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt).to(torch.float16)
    # Create our random starting point
    latents = torch.randn((2, 4, 64, 64), device=device, generator=generator).to(torch.float16)
    latents *= pipe.scheduler.init_noise_sigma

    # Prepare the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    k = pipe.scheduler.timesteps
    start_time=time.time()
    # first stage text2img
    latents = training(pipe.scheduler.timesteps, pipe, latents, cond_fn, text_embeddings)
    print(time.time()-start_time)

    with torch.no_grad():
        #image = pipe.vae.decode(latents.detach() / 0.18215).sample
        image = pipe.decode_latents(latents.detach())

    image = (image[0] * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image = image.resize((256, 256), resample=PIL.Image.LANCZOS)
    subdir_name = "re0"
    subdir_path = os.path.join(args.result_dir, subdir_name)

    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

    image.save(os.path.join(subdir_path, file_name))

    # second stage img2img
    init_seed(args.seed)
    img_pipe.scheduler.set_timesteps(img_num_inference_steps, device=device)
    timesteps,_ = img_pipe.get_timesteps(img_num_inference_steps, strength, device)
    latent_timestep = timesteps[:1]
    #generate noise
    shape = latents.shape

    for i in range(args.iter):
        noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32, layout=torch.strided).to(device)
        latents1 = img_pipe.scheduler.add_noise(latents[0:1], noise[0:1], latent_timestep).to(torch.float16)
        latents2 = img_pipe.scheduler.add_noise(latents[1:2], noise[1:2], latent_timestep).to(torch.float16)
        latents = torch.cat([latents1, latents2])
        latents = training(timesteps, img_pipe, latents, cond_fn, text_embeddings)

        with torch.no_grad():
        #image = pipe.vae.decode(latents.detach() / 0.18215).sample
            image = pipe.decode_latents(latents.detach())

        image = (image[0] * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize((256, 256), resample=PIL.Image.LANCZOS)
        subdir_name = f"re{i+1}"
        subdir_path = os.path.join(args.result_dir, subdir_name)

        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

        image.save(os.path.join(subdir_path, file_name))

    print(time.time()-start_time)

    """ # Decode the resulting latents into an image
    with torch.no_grad():
        #image = pipe.vae.decode(latents.detach() / 0.18215).sample
        image = pipe.decode_latents(latents.detach())

    image = (image[0] * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image = image.resize((256, 256), resample=PIL.Image.LANCZOS)
    image.save(os.path.join(args.result_dir, file_name)) """