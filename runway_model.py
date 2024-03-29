# https://colab.research.google.com/drive/1RUaVUqCvyojwoMglp6cFoLDnCfLHBZtB#scrollTo=PVYrjvgE_8AU

import helpers
import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import matplotlib.pyplot as plt
import runway

latent_vector_1=0
latent_vector_2=0

@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
	global Gs
	tflib.init_tf()
	model = opts['checkpoint']
	print("open model %s" % model)
	with open(model, 'rb') as file:
		G, D, Gs = pickle.load(file)
	Gs.print_layers()
	# load latent representation
	p1 = inputs['people_vector1']
	global latent_vector_1
	latent_vector_1 = np.load("latent_representations/j_01.npy")
	p2 = inputs['people_vector2']
	global latent_vector_2
	latent_vector_2 = np.load("latent_representations/j_02.npy")
	global generator
	generator = Generator(Gs, batch_size=1, randomize_noise=False)
	return generator

# def setup(opts):
# 	tflib.init_tf()
# 	url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
# 	with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
# 		_G, _D, Gs = pickle.load(f)
# 		# _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
# 		# _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
# 		# Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
# 	Gs.print_layers()
# 	global generator
# 	generator = Generator(Gs, batch_size=1, randomize_noise=False)
# 	return Gs

def generate_image(generator, latent_vector):
	latent_vector = latent_vector.reshape((1, 18, 512))
	generator.set_dlatents(latent_vector)
	img_array = generator.generate_images()[0]
	img = PIL.Image.fromarray(img_array, 'RGB')
	return img.resize((512, 512))   

generate_inputs = {
	'age': runway.number(min=0, max=6, default=3, step=0.1),
}

generate_outputs = {
	'image': runway.image(width=512, height=512),
}

@runway.command('babey generator', inputs=generate_inputs, outputs=generate_outputs)
def move_and_show(model, inputs):	
	global latent_vector_1
	global latent_vector_2
	latent_vector = (latent_vector_1 + latent_vector_2) / 2
	# load direction
	age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
	direction = age_direction
	# model = generator
	coeff = 7 - inputs['age']
	new_latent_vector = latent_vector.copy()
	new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
	image = generate_image(model, new_latent_vector)
	#ax[i].set_title('Coeff: %0.1f' % coeff)
	#plt.show()
	return {'image': image}

if __name__ == '__main__':
	runway.run(debug=True)