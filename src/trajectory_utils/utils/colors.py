from __future__ import annotations

import colorsys
import torch

def get_colormap(base_color = "ff0000"):
    def time_color_funnc(t: torch.Tensor) -> list[str]:
        # Smooth color gradient 
        max_value = t.max()
        min_value = t.min()
        t = (t - min_value) / (max_value - min_value)
        color = [int(base_color[i:i+2], 16)/255 for i in (0, 2, 4)]
        import colorsys
        base_color_hsv = colorsys.rgb_to_hsv(*color)
        colors = []
        for progress in t:
            h,s,v = base_color_hsv
            h = h + progress*0.4
            s = s #+ torch.sin(progress*3.14)*0.1
            color = colorsys.hsv_to_rgb(h, s, v)
            colors.append(color)
        return [f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}" for color in colors]
    return time_color_funnc
