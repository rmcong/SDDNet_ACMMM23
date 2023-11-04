import torch
from torchvision.utils import draw_segmentation_masks

def colorize_classid_array(classid_array, image=None, alpha=0.8, colors=None):
    """
    Args:
        classidx_array: torch.LongTensor, (H, W) tensor
        num_cls: int, number of classes
        image: if None, overlay colored label on it, otherwise a pure black image is created
        colors: list/dict/array provdes class id to color mapping
    """
    if image is None:
        image = torch.zeros(size=(3, classid_array.size(-2), classid_array.size(-1)),
                            dtype=torch.uint8)
    # if colors is not None:
    #     assert len(colors) == num_cls, 'size of colormap should be consistent with num_cls'
    # all_class_masks = (classid_array == torch.arange(num_cls)[:, None, None])
    # im_label_overlay = draw_segmentation_masks(image, all_class_masks, alpha=alpha, colors=colors)
    unique_idx = torch.unique(classid_array)
    colors_use = [colors[idx.item()] for idx in unique_idx]
    all_class_masks = (classid_array == unique_idx[:, None, None])
    im_label_overlay = draw_segmentation_masks(image, all_class_masks, alpha=alpha, colors=colors_use)

    return im_label_overlay, unique_idx