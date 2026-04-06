import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

json_files = ['C:/Projects/ESE5971/reports/rank8_lr1e5.json', 
              'C:/Projects/ESE5971/reports/rank8_lr2e5.json', 
              'C:/Projects/ESE5971/reports/rank16_lr1e5.json', 
              'C:/Projects/ESE5971/reports/rank16_lr2e5.json']

images = [
    'C:/Projects/ESE5971/reports/rank8_lr1e5.png', 
    'C:/Projects/ESE5971/reports/rank8_lr2e5.png',
    'C:/Projects/ESE5971/reports/rank16_lr1e5.png', 
    'C:/Projects/ESE5971/reports/rank16_lr2e5.png'
]

titles = [
    'Rank: 8, Learning Rate: 1e-5',
    'Rank: 8, Learning Rate: 2e-5',
    'Rank: 16, Learning Rate: 1e-5',
    'Rank: 16, Learning Rate: 2e-5'
]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, ax in enumerate(axes.flat):
    img = mpimg.imread(images[i])
    ax.imshow(img)
    ax.axis('off')  
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

for file_name in json_files:
    with open(file_name, 'r') as f:
        data = json.load(f)
    
    rank = data.get('rank')
    lr = data.get('learning_rate')
    final_train = data['train'][-1]['loss']
    final_eval = data['eval'][-1]['loss']
    
    print(f"Rank {rank}, LR {lr}")
    print(f"  - Final Train Loss: {final_train:.4f}")
    print(f"  - Final Eval Loss:  {final_eval:.4f}")
    print("-" * 30)