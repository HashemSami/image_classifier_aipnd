import torch
from time import time, sleep
from model import load_model, save_checkpoint
from dataloader import dataloader
from torch import optim, nn
from helpers import get_train_input_args

def main():

    in_arg = get_train_input_args()

    train(in_arg.dir, 
          in_arg.arch, 
          in_arg.learning_rate, 
          in_arg.hidden_units, 
          in_arg.epochs,
          in_arg.checkpoint, 
          in_arg.gpu)

# train the model and save the wieghts in checkpoint
def train(data_dir, arch, learning_rate, hidden_units, epochs, checkpoint, gpu):

    start_time = time()

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    loader, datasets = dataloader(data_dir)

    model = load_model(arch, checkpoint, hidden_units)

    model.to(device)

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    steps = 0
    running_loss = 0
    print_every = 10
    
    for epoch in range(epochs):
        for images, labels in loader["trainloder"]:
            steps+=1
            #move images and lable to the default device
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item()
        
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
        
                with torch.set_grad_enabled(False):
                    validationloader = loader["validationloader"]
                    for images, labels in validationloader:
                        images, labels = images.to(device), labels.to(device)
        
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()
        
                        # calculate accuracy
                        ps = logps.exp()
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"validation loss: {validation_loss/len(validationloader):.3f}.. "
                    f"validation accuracy: {accuracy/len(validationloader):.3f}")

                running_loss = 0
                model.train()

    save_checkpoint(arch, model, hidden_units, optimizer, epochs, datasets["train"])

    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

if __name__ == "__main__":
    main()