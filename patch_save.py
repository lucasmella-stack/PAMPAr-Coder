#!/usr/bin/env python3
"""Patch train_cloud.py to save lighter checkpoints."""
with open('/workspace/PAMPAr-Coder/cloud/runpod/train_cloud.py', 'r') as f:
    content = f.read()

# Replace the torch.save block
old_save = "            'optimizer': optimizer.state_dict(),\n            'scheduler': scheduler.state_dict(),\n"

# We'll just remove optimizer/scheduler from default saves
# and add a conditional
old_block = """        torch.save({
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
        }, path)"""

new_block = """        # Save optimizer only every 1000 steps or at epoch end (saves ~14GB per checkpoint)
        save_optimizer = (self.global_step % 1000 == 0) or ('epoch' in str(filename))
        
        checkpoint_data = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
        }
        
        if save_optimizer:
            checkpoint_data['optimizer'] = optimizer.state_dict()
            checkpoint_data['scheduler'] = scheduler.state_dict()
            print(f"  (full checkpoint with optimizer)")
        
        torch.save(checkpoint_data, path)"""

if old_block in content:
    content = content.replace(old_block, new_block)
    with open('/workspace/PAMPAr-Coder/cloud/runpod/train_cloud.py', 'w') as f:
        f.write(content)
    print("PATCHED OK: checkpoints now ~6.5GB instead of ~20GB")
else:
    print("ERROR: target block not found - may already be patched")
    # Check if already patched
    if 'save_optimizer' in content:
        print("Already patched!")
