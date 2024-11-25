'''
powershell -ExecutionPolicy Bypass -File "C:\Users\edoar\Desktop\PythonProjects\deep-learning-application\Lab4\ood_detection\run.ps1"
'''                                                                           
# python train.py --batch_size 256 --epochs 10 --ood_set fakedata 
# python train.py --batch_size 256 --epochs 10 --ood_set cifar100 
# python train.py --batch_size 256 --epochs 50 --ood_set fakedata 
# python train.py --batch_size 256 --epochs 50 --ood_set cifar100 

# python eval.py --batch_size 256  --ood_set fakedata --pretrained './checkpoints/cifar10_cnn_10.pth'
# python eval.py --batch_size 256  --ood_set cifar100 --pretrained './checkpoints/cifar10_cnn_10.pth'
# python eval.py --batch_size 256  --ood_set fakedata --pretrained './checkpoints/cifar10_cnn_50.pth'
# python eval.py --batch_size 256  --ood_set cifar100 --pretrained './checkpoints/cifar10_cnn_50.pth'

python eval.py --batch_size 256  --ood_set fakedata --pretrained checkpoints/cifar10_cnn_2.pth