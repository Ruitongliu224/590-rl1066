
#--------------------------------------
#EXAMPLE
#--------------------------------------

#RUN --> python KERAS-VIZ.py ; xdg-open model_plot.png  &
#NOTE: "none" in the shape means it does not have a pre-defined number. 
#For example, it can be the batch size you use during training, and 
#you want to make it flexible by not assigning any value to it so that
# you can change your batch size.

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

model = Sequential()
model.add(Dense(3, activation='sigmoid', input_shape=(2,))) #none
model.add(Dense(2, activation='sigmoid')) #none
model.add(Dense(1, activation='linear')) #none
print(model.summary())
plot_model(
model,
to_file='model_plot.png',
show_shapes=True, 
show_layer_names=True)

layer=model.get_layer( name='dense')
print(layer.name,layer.get_weights())
print(layer.weights)

