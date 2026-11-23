"""Optional frozen YOLOv8 detector as a source of category-agnostic ROI evidence."""
import torch
from torch import nn
from torchvision.ops import box_convert, nms, roi_align

from .anchor_context import VIEWS, crop, resize, window


class FrozenProposer:
    """Keep the detector outside the trainable reader and its checkpoint."""
    def __init__(self, path, device, confidence=.05, top_k=6):
        from ultralytics import YOLO
        detector = YOLO(str(path))
        if detector.task != "detect":
            raise ValueError("Expected a YOLOv8 detection checkpoint")
        self.model = detector.model.to(device).float().eval().requires_grad_(False)
        self.head = self.model.model[-1]
        if self.head.__class__.__name__ != "Detect" or getattr(self.head, "end2end", False):
            raise ValueError("Expected the standard YOLOv8 Detect head")
        self.device, self.confidence, self.top_k = device, confidence, top_k
        self.channels = self.head.cv2[0][0].conv.in_channels

    @torch.no_grad()
    def __call__(self, view, origin, extent, image_hw):
        # Copy the first input reference before Detect replaces list elements.
        features = []
        handle = self.head.register_forward_pre_hook(lambda _, args: features.append(args[0][0]))
        try:
            output = self.model(view.to(self.device))
        finally:
            handle.remove()
        pred = output[0] if isinstance(output, tuple) else output
        if pred.ndim != 3 or pred.shape[1] != 4 + self.head.nc or len(features) != 1:
            raise RuntimeError("Unexpected YOLO decoded predictions or pyramid features")
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Non-finite YOLO predictions")
        height, width = view.shape[-2:]
        origin_y, origin_x = origin
        extent_h, extent_w = extent
        # Decode/clamp in normalized view coordinates. Exclude replicated padding.
        detections = pred[0].T
        boxes = box_convert(detections[:, :4], "cxcywh", "xyxy")
        boxes = boxes / boxes.new_tensor([width,height,width,height])
        left, top = max(0., -origin_x/extent_w), max(0., -origin_y/extent_h)
        right = min(1., (image_hw[1]-origin_x)/extent_w)
        bottom = min(1., (image_hw[0]-origin_y)/extent_h)
        boxes[:, 0::2].clamp_(left, right)
        boxes[:, 1::2].clamp_(top, bottom)
        scores = detections[:, 4:].amax(1)
        valid = (scores >= self.confidence) & (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])
        boxes, scores = boxes[valid], scores[valid]
        chosen = nms(boxes, scores, .5)[:self.top_k]
        boxes, scores = boxes[chosen], scores[chosen]
        f = features[0]
        hf, wf = f.shape[-2:]
        pooled = roi_align(f, [boxes * boxes.new_tensor([wf,hf,wf,hf])], (2,2),
                           aligned=True, sampling_ratio=2).mean((-2,-1))
        native = boxes * boxes.new_tensor([extent_w,extent_h,extent_w,extent_h])
        native += boxes.new_tensor([origin_x,origin_y,origin_x,origin_y])
        return pooled, native, scores


class ProposalReader(nn.Module):
    def __init__(self, proposer, classes, pooling="learned"):
        super().__init__()
        self.proposer = proposer
        self.pooling = pooling
        self.embed = nn.Sequential(nn.Linear(proposer.channels + 5, 24), nn.ReLU())
        self.scale = nn.Embedding(4, 24)
        self.relevance = nn.Sequential(nn.Linear(24 + classes, 24), nn.ReLU(), nn.Linear(24, 1))

    def acquire(self, image, tile, name, cache):
        if name == "scene" and "scene_proposals" in cache:
            return cache["scene_proposals"]
        hw = image.shape[-2:]
        if name == "scene":
            view = resize(image[None], (160,160))
            origin, extent = (0,0), hw
        else:
            footprint, side = (640,512) if name == "local" else VIEWS[name]
            view = resize(crop(image[None], tile, footprint), (side,side))
            origin, extent = window(tile, footprint), (footprint,footprint)
        result = self.proposer(view, origin, extent, hw)
        count = len(result[1])
        cache["yolo_calls"] = cache.get("yolo_calls",0) + 1
        cache["yolo_rois"] = cache.get("yolo_rois",0) + count
        cache["yolo_empty"] = cache.get("yolo_empty",0) + int(count == 0)
        if name == "scene":
            cache["scene_proposals"] = result
        return result

    def forward(self, image, tile, sources, local, cache):
        vectors = []
        y,x,h,w = tile
        for name in ("local", *sources):
            feat, native, score = self.acquire(image, tile, name, cache)
            if len(score):
                relative = (native - native.new_tensor([x,y,x,y])) / native.new_tensor([w,h,w,h])
                vectors.append(self.embed(torch.cat((feat, relative, score[:,None]), 1))
                               + self.scale.weight[("local","medium","large","scene").index(name)])
        if not vectors:
            return local.new_zeros((1,24))
        values = torch.cat(vectors)
        if self.pooling == "uniform":
            return values.mean(0,keepdim=True)
        anchor = local.softmax(1).mean((-2,-1)).expand(len(values),-1)
        scores = self.relevance(torch.cat((values,anchor),1)).flatten()
        # Null option has score zero and a zero contribution.
        weights = torch.cat((scores, scores.new_zeros(1))).softmax(0)[:-1]
        return (values * weights[:,None]).sum(0,keepdim=True)
