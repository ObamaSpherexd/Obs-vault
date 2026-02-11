Colab is used to perform tasks on the external PC to preserve resources
### Uploading files to Colab
To work with files to Colab we first need to upload them to the cloud
!**We need to upload files every time we start Colab as it has buffer memory**!
1. Press the following button on the left panel![[Pasted image 20251205143646.png]]
2. Then we select 'Upload files'![[Pasted image 20251205143711.png]]
3. Select a file and upload it

### Mounting google disk in colab
```python
from google.colab import drive
drive.mount('/content/gdrive')

import os
os.chdir(r'/content/gdrive/MyDrive/Digital department ML/Pandas lecture')
os.listdir() # ['dataset.csv', 'data_regex.csv']
```
