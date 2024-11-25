'''
powershell -ExecutionPolicy Bypass -File "C:\Users\edoar\Desktop\PythonProjects\deep-learning-application\Lab4\fgsm\run.ps1"
'''  
# python fgsm.py --epsilon 0.01 --batch_size 256 --pretrained C:\Users\edoar\Desktop\PythonProjects\deep-learning-application\Lab4\ood_detection\checkpoints\cifar10_cnn_2.pth

# python eval.py --batch_size 256 --pretrained C:\Users\edoar\Desktop\PythonProjects\deep-learning-application\Lab4\ood_detection\checkpoints\cifar10_cnn_2.pth
# python adv_training.py --batch_size 256 --epsilon 0.01 --epochs 2
python eval_robust_cnn.py --batch_size 256 --robust-model C:\Users\edoar\Desktop\PythonProjects\deep-learning-application\Lab4\fgsm\checkpoints\robust_cnn_2.pth
