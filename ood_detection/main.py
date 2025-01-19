import torch
import os
import argparse
from models import get_model
from data import get_data_loaders
from train import train_model, evaluate_model
from utils import (compute_scores, max_softmax, plot_distributions, 
                  plot_logits_softmax, plot_sorted_scores, plot_roc_pr_curves,
                  calculate_metrics, ensure_results_dir, save_model, load_model)

def main():
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description='OOD Detection')
    parser.add_argument('--train', action='store_true', 
                       help='Train the model if this flag is provided')
    parser.add_argument('--model-type', type=str, choices=['cnn', 'resnet'],
                       default='cnn', help='Type of model to use (default: cnn)')
    parser.add_argument('--model-path', type=str, 
                       help='Path to the model file')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Training epochs')
    args = parser.parse_args()

    # Create results directory
    ensure_results_dir()
    
    # Set device and print GPU info if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("Using CPU")
    
    # Get data loaders
    trainloader, testloader, oodloader = get_data_loaders(batch_size=256)
    
    # Initialize model
    print(f"\nInitializing {args.model_type.upper()} model...")
    model = get_model(args.model_type).to(device)
    
    # Define model name based on type and epochs
    if args.model_type == "cnn":
        model_name = f"{args.model_type}_model_ep{args.epochs}"
    if args.model_type == "resnet":
        model_name = f"{args.model_type}"
   
    model_path = args.model_path or f'results/{model_name}'
    
    # Train or load model based on arguments
    if args.train:
        print("\nTraining model...")
        epochs = args.epochs
        model = train_model(model, trainloader, epochs=epochs, device=device)
        
        # Save the trained model
        accuracy = evaluate_model(model, testloader, device)
        save_model(model, epochs, accuracy, model_path)
        print(f"\nModel saved to {model_path}")
    else:
        print("\nLoading existing model...")
        try:
            model, checkpoint = load_model(model, model_path, device)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please train the model first using --train flag or provide correct model path")
            return
        
        # Evaluate loaded model
        print("\nEvaluating loaded model...")
        evaluate_model(model, testloader, device)

    # Compute scores for ID and OOD data
    print("\nComputing scores for ID and OOD data...")
    scores_test = compute_scores(testloader, max_softmax, model, device)
    scores_ood = compute_scores(oodloader, max_softmax, model, device)
    
    # Generate all plots and metrics
    print("\nGenerating visualizations and metrics...")
    plot_distributions(scores_test, scores_ood, title=model_name)
    plot_sorted_scores(scores_test, scores_ood, title=model_name)
    plot_roc_pr_curves(scores_test, scores_ood, title=model_name)
    
    # Calculate and display metrics
    metrics = calculate_metrics(scores_test, scores_ood)
    
    # Sample image analysis
    print("\nAnalyzing sample image...")
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    sample_image = images[0]
    true_label = labels[0].item()
    print(f"True label: {true_label}")
    plot_logits_softmax(model, sample_image, device, title=model_name)
    
    # Print final GPU memory usage if available
    if device.type == 'cuda':
        print(f"\nFinal Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
