import jax
import jax.numpy as jnp
import json
import numpy as np

f32 = jnp.float32

class AugmentationProcessor:
  def __init__(self, augmentations_config, imgkeys):
    self.augmentations_config = augmentations_config
    self.augmentations = {}
    self.augmentations_probs = None
    self.imgkeys = imgkeys

  def create_augmentations(self, obs_space):
    augmentation_names_kwargs = getattr(
      self.augmentations_config,
      "encoder_augmentations",
      ['{"bounding_box": {"crop_rate": 2, "upsample": True}}']
    )
    augmentation_names_kwargs = [json.loads(x) for x in augmentation_names_kwargs]
    N_augmentations = len(augmentation_names_kwargs)

    print(F"TRAINING WITH {N_augmentations} AUGMENTATIONS: {augmentation_names_kwargs}")

    self.augmentations = {}
    for imgkey in self.imgkeys:
      H, W, C = obs_space[imgkey].shape[-3:]
      self.augmentations[imgkey] = []
      for i in range(N_augmentations):
        for name, kwargs in augmentation_names_kwargs[i].items():
          self.augmentations[imgkey].append(self.get_aug(H, W, C, i, name, kwargs))
    
    if not getattr(self.augmentations_config, "concat_augmentations", False):
      self.augmentation_probs = {}
      for imgkey in self.imgkeys:
        if hasattr(self.augmentations_config, "augmentations_probs"):
          self.augmentations_probs[imgkey] = self.augmentations_config.augmentations_probs
        else:
          self.augmentations_probs[imgkey] = [1 / (N_augmentations + 1) for _ in range(N_augmentations)]

    return self.augmentations
  
  def _stack_augmentations(self, imgs_k_aug, imgs_k):
    n_stack_by_axis = self.augmentations_config.n_stack_by_axis
    current_stack = []
    imgs_k_aug_stacked = []

    H, W, _ = imgs_k.shape[-3:]
    for aug in imgs_k_aug:
      h, w, _ = aug.shape[-3:]
      assert (h * n_stack_by_axis == H) and (w * n_stack_by_axis == W), \
        f"Attempting to stack by each axis {n_stack_by_axis} augmentations with shapes ({h}, {w}), while the final image has shape ({H}, {W})"
      
      current_stack.append(aug)
      if len(current_stack) == n_stack_by_axis * n_stack_by_axis:
          img_stacked = []
          for i in range(n_stack_by_axis):
            img_stacked.append(jnp.concatenate([
              current_stack[j]
              for j in range(i * n_stack_by_axis, (i + 1) * n_stack_by_axis)
            ], axis=-2))
          img_stacked = jnp.concatenate(img_stacked, axis=-3)
          imgs_k_aug_stacked.append(img_stacked)
          current_stack = []

    assert len(current_stack) == 0, \
        f"Attempting to stack {len(imgs_k_aug)} augmentations, {n_stack_by_axis * n_stack_by_axis} per each image."
    
    return imgs_k_aug_stacked

  def _expand_aug_axis(self, obs):
    imgs = {k: obs[k] for k in sorted(self.imgkeys)}
    for k, imgs_k in imgs.items():
      aug_axis = len(imgs_k.shape) - 3
      obs[k] = jnp.expand_dims(imgs_k, axis=aug_axis)
    return obs      
  
  def _apply_augmentations(self, obs):
    imgs = {k: obs[k] for k in sorted(self.imgkeys)}

    for k, imgs_k in imgs.items():
      if getattr(self.augmentation_config, "concat_augmentations", True):
        imgs_k_aug = [aug(imgs_k) for aug in self.augmentations[k]]
        # otherwise all augmentations must be upscaled
        if getattr(self.augmentations_config, "stack_augmentations", False):
          imgs_k_aug = self._stack_augmentations(imgs_k_aug, imgs_k)

        if getattr(self.augmentations_config, "use_inital_image", True):
          imgs_k_aug = [imgs_k] + imgs_k_aug
      else:
        aug = np.random.choice(self.augmentations)
        imgs_k_aug = aug

        if getattr(self.augmentations_config, "use_inital_image", True):
          imgs_k_aug = [imgs_k] + imgs_k_aug
        
      if len(imgs_k_aug) > 1 or getattr(self.augmentations_config, "expand_aug_axis", True):
        aug_axis = len(imgs_k.shape) - 3
        imgs_k_aug = [jnp.expand_dims(c, axis=aug_axis) for c in imgs_k_aug]
      
      if len(imgs_k_aug) > 1:
        imgs_k_aug = jnp.concatenate(imgs_k_aug, axis=aug_axis)

      print(f"Changed shape from {imgs_k.shape} to {imgs_k_aug.shape}")
      
      obs[k] = imgs_k_aug
    return obs
  
  @staticmethod
  def get_aug_bounding_box(H, W, aug_ind, fixed=True, y=None, x=None, crop_rate=2, crop=True, upsample=False):
    crop_h, crop_w = H // crop_rate, W // crop_rate

    # sample x, y only on augmentation function creation
    if fixed:
      if y is None:
        y = np.random.randint(0, H - crop_h + 1)
      if x is None:
        x = np.random.randint(0, W - crop_w + 1)

    print(f"Augmentation {aug_ind + 1} is an upscaled crop of shape ({crop_h}, {crop_w}) starting from ({y}, {x}), original image shape: ({H}, {W})")
  
    def aug(imgs, y=y, x=x, ch=crop_h, cw=crop_w):
      if y is None:
        y = np.random.randint(0, H - crop_h + 1)
      if x is None:
        x = np.random.randint(0, W - crop_w + 1)
      
      if crop:
        imgs_new = imgs[..., y:y+ch, x:x+cw, :]
      else:
        imgs_new = jnp.zeros_like(imgs)
        imgs_new[..., y:y+ch, x:x+cw, :] = imgs[..., y:y+ch, x:x+cw, :]

      if not upsample:
        return imgs_new
      
      # resize back to (H, W)
      # jax.image.resize wants floats, so cast to f32, resize, then back to uint8
      imgs_new_f = imgs_new.astype(f32) / 255.0
      out = jax.image.resize(imgs_new_f, imgs.shape, method='bilinear')
      out = (out * 255).astype(np.uint8)
      return out

    return aug

  @staticmethod
  def get_aug_fixed_half_bounding_box(H, W, aug_ind, y=None, x=None, crop=True, upsample=False):
    crop_h, crop_w = H // 2, W // 2

    # Computes predetermined positions if not provided with y and x:
    # [1] [2]
    # [3] [4]
    if y is None:
      y = ((aug_ind % 4) // 2) * (H // 2)
    if x is None:
      x = ((aug_ind % 4) % 2) * (W // 2)
  
    def aug(imgs, y=y, x=x, ch=crop_h, cw=crop_w):
      if crop:
        imgs_new = imgs[..., y:y+ch, x:x+cw, :]
      else:
        imgs_new = jnp.zeros_like(imgs)
        imgs_new[..., y:y+ch, x:x+cw, :] = imgs[..., y:y+ch, x:x+cw, :]

      if not upsample:
        return imgs_new

      imgs_new_f = imgs_new.astype(f32) / 255.0
      out = jax.image.resize(imgs_new_f, imgs.shape, method='bilinear')
      out = (out * 255).astype(np.uint8)
      return out
    return aug

  @staticmethod
  def get_aug_random_shift(H, W, aug_ind, fixed=True, y=None, x=None, pad=4):
    # Pads input images and then randomly crops the inital image size

    if fixed:
      if y is None:
        y = np.random.randint(0, 2 * pad + 1)
      if x is None:
        x = np.random.randint(0, 2 * pad + 1)
    
    def aug(imgs, y=y, x=x, H=H, W=W, pad=pad):
      if y is None:
        y = np.random.randint(0, 2 * pad + 1) # Not fixed = sample new each time
      if x is None:
        x = np.random.randint(0, 2 * pad + 1)
      
      nd = imgs.ndim
      pad_width = [(0, 0)] * nd
      pad_width[-3] = (pad, pad)
      pad_width[-2] = (pad, pad)

      imgs_padded = jnp.pad(imgs, pad_width, mode='constant', constant_values=0)
      imgs_new = imgs_padded[..., y:y+H, x:x+W, :]
      return imgs_new

    return aug

  @staticmethod
  def get_aug_horizontal_flip(H, W, aug_ind):
    def aug(imgs):
      return jnp.flip(imgs, axis=-2)
    return aug

  @staticmethod
  def get_aug_vertical_flip(H, W, aug_ind):
    def aug(imgs):
      return jnp.flip(imgs, axis=-3)
    return aug
    
  def get_aug(self, H, W, C, aug_ind, aug_name, aug_kwargs):
    get_aug_by_name = getattr(self, f"get_aug_{aug_name}", None)
    assert get_aug_by_name is not None, f"Augmentation {aug_name} not found!"
    
    aug = get_aug_by_name(H, W, aug_ind, **aug_kwargs)
    # print("AUG Function:", type(aug), aug)
    return aug
