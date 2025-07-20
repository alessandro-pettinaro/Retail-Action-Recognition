import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from VideoMamba.videomamba.video_sm.models.videomamba import VisionMamba, _cfg, load_state_dict as load_videomamba_weights
import os 


MODEL_PATH = '/home/vrai/video-mamba/pretrain'
_MODELS = {
"videomamba_m16_k400_mask_ft_f16_res224": os.path.join(MODEL_PATH, "videomamba_m16_k400_mask_ft_f16_res224.pth"),
}

class VideoMambaWrapper(nn.Module):

    """
    Wrapper del modello VideoMamba per classificazione video,
    con supporto per fine-tuning e stampa dello stato dei layer.
    """

    def __init__(self,load_pretrained=False,apply_finetune=False,load_checkpoint=False,get_checkpoint_path=None,train_last_layers=0,num_class=0):
        super().__init__()

        self.apply_finetune=apply_finetune
        self.load_pretrained=load_pretrained
        self.load_checkpoint=load_checkpoint
        self.get_checkpoint_path=get_checkpoint_path
        self.train_last_layers=train_last_layers

        # Carica tutto il modello (backbone + head)
        self.backbone = self.videomamba_middle(pretrained=load_pretrained,checkpoint=load_checkpoint,checkpoint_path=get_checkpoint_path,num_class=num_class)
        
        self.apply_finetune_strategy()
    


    def forward(self, x):
        return self.backbone(x)

    @staticmethod
    def remove_prefix_from_state_dict(state_dict, prefix="backbone."):
        """
        Rimuove un prefisso da tutte le chiavi dello state_dict (utile per i checkpoint completi).
        """
        return {
            k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()
    }

    @staticmethod
    def videomamba_middle(pretrained=False, checkpoint=False, checkpoint_path=None,num_class=0, **kwargs):

        model = VisionMamba(
            patch_size=16, 
            embed_dim=576,
            drop_rate=0.2, 
            depth=32,
            num_classes=num_class,
            num_frames=16,
            drop_path_rate=0.5, 
            rms_norm=True, 
            residual_in_fp32=True, 
            fused_add_norm=True,
            **kwargs
        )
        model.default_cfg = _cfg()
        print(model)

        if  pretrained:
            print("[Info] Caricamento pesi pre-addestrati standard videomamba_m16_k400_mask_ft_f16_res224 ")
            state_dict = torch.load(_MODELS["videomamba_m16_k400_mask_ft_f16_res224"], map_location='cpu')
            load_videomamba_weights(model, state_dict, center=True)
        if checkpoint:
            model.head
            print("[Info] Caricamento checkpoint fine-tuned")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            state_dict = VideoMambaWrapper.remove_prefix_from_state_dict(state_dict, prefix="backbone.")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[Checkpoint] Pesi caricati. Mancanti: {missing}, Inattesi: {unexpected}") 


        return model  



    def apply_finetune_strategy(self):
        """
        Applica la strategia di fine-tuning:
        - Se disattivato: congela tutto (backbone + head)
        - Se attivato: sblocca gli ultimi N blocchi + la testa
        """
        if not self.apply_finetune:
            print("[Info] Fine-tuning disabilitato. Congelo tutto il backbone e la testa.")
            for param in self.backbone.parameters():
                param.requires_grad = False
            return

        print(f"[Info] Fine-tuning attivo: sblocco ultimi {self.train_last_layers} blocchi e la testa")

        # Congela tutto inizialmente
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Sblocca ultimi N blocchi (es. backbone.layers[-N:])
        if self.train_last_layers > 0 and hasattr(self.backbone, "layers"):
            for block in self.backbone.layers[-self.train_last_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Sblocca la testa
        if hasattr(self.backbone, "head"):
            for param in self.backbone.head.parameters():
                param.requires_grad = True

    
    def load_custom_checkpoint(self, checkpoint_path, device='cpu', strict=False):
        """
        Carica i pesi del modello da un checkpoint.

        Args:
            checkpoint_path (str): Percorso del file .pt o .pth contenente lo state_dict
            device (str or torch.device): Dove mappare i tensori ('cpu' o 'cuda')
            strict (bool): Se True, richiede corrispondenza esatta dei layer
        """
        print(f"[Checkpoint] Caricamento da: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(state_dict, strict=strict)
        print("[Checkpoint] Pesi caricati con successo.")

