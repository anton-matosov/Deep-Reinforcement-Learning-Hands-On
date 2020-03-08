import math
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}

    with SummaryWriter() as writer:
        for angle in range(-360, 360):
            angle_rad = angle * math.pi / 180
            for name, fun in funcs.items():
                val = fun(angle_rad)
                if val != float("nan") and val != float("InF") and abs(val) < 1000:
                    writer.add_scalar(name, val, angle)

