import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch Siamese Network for Radar Signal Classification')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args, unknown = parser.parse_known_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # Load datasets
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    num_classes = 2# Or any other number you choose
    # train_dataset, test_dataset = load_and_split_data('GOLD_XYZ_OSC.0001_1024.hdf5', num_classes=num_classes)
    # Get the directory of the main.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(current_dir)
    # Construct the full path to the data file
    data_file_path = os.path.join(project_root, 'GOLD_XYZ_OSC.0001_1024.hdf5')
    train_dataset, test_dataset = load_and_split_data(data_file_path, num_classes=num_classes)
    print("Train set distribution:")
    print_class_distribution(train_dataset)
    print("\nTest set distribution:")
    print_class_distribution(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    siamesemodel = SiameseNetwork(num_classes).to(device)
    optimizer = optim.Adadelta(siamesemodel.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        siamesemodel.train_epoch(args, siamesemodel, device, train_loader, optimizer, epoch)
        siamesemodel.test(siamesemodel, device, test_loader)
        # Generate and visualize embeddings after each epoch
        # embeddings, labels = get_embeddings(model, test_loader, device)
        # visualize_embeddings(embeddings, labels, f'Test Embeddings (t-SNE) - Epoch {epoch}', save_path=f'embeddings_epoch_{epoch}.png')
        scheduler.step()

    if args.save_model:
        torch.save(siamesemodel.state_dict(), "siamese_radar_signal.pt")

if __name__ == '__main__':
    main()
