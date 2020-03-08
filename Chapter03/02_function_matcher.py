import torch
import torch.nn as nn
import torch.optim as optim
import math
from tensorboardX import SummaryWriter



class TrigonometryModule(nn.Module):
    def __init__(self, num_inputs = 1, num_outputs = 1, dropout_prob=0.3):
        super(TrigonometryModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 15),
            nn.ReLU(),
            # nn.Dropout(p=dropout_prob),
            nn.Linear(15, 25),
            nn.ReLU(),
            nn.Linear(25, num_outputs),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)

if __name__ == "__main__":
    num_inputs = 1
    num_epochs = 50000
    batch_size = 32
    report_every_steps = 100

    net = TrigonometryModule(num_inputs=num_inputs)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    global_step = 0
    with SummaryWriter() as writer:
      # writer.add_hparams()
      epoch = 0
      while epoch < num_epochs:
        epoch += 1
        inputs = torch.ones(batch_size, num_inputs).normal_(0, 2 * math.pi)
        expected_outputs = torch.sin(inputs)

        optimizer.zero_grad()

        out = net(inputs)
        loss = loss_func(out, expected_outputs)
        loss.backward()
        optimizer.step()
        
        global_step += 1
        if global_step % report_every_steps == 0:
          writer.add_scalar('loss', loss, global_step)
          writer.add_histogram('out', out, global_step)
          writer.add_histogram('expected_outputs', expected_outputs, global_step)
          print(loss)

