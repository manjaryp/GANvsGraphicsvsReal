# SC-EffNet


For SC-EffNet a sample program with XXX color space is given in [pgm.py](#). 


1. If you wanted to change the color space (e.g. SC-EffNet<sub>RGB</sub>, SC-EffNet<sub>HSV</sub>, SC-EffNet<sub>LCH</sub>, etc.)do the folowing changes.


```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('path to accuracy plot/acc.png')
plt.show()
```


2. for  non sclaed version 

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('path to accuracy plot/acc.png')
plt.show()
```

For SC-EffNet<sub>RGB</sub>: [Link](#)
