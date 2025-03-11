![oreilly-logo](images/oreilly.png)

# Deep Learning for Modern AI


This repository contains code for the [O'Reilly Live Online Training for Deep Learning for Modern AI](https://learning.oreilly.com/live-events/deep-learning-for-modern-ai/0642572005084)

This training provides the theory and practical concepts for a comprehensive introduction to machine learning and deep learning with PyTorch —foundational knowledge needed to successfully build and train GenAI and multimodal models. By making our way through several real-world case studies including object recognition and text classification this session is an excellent crash course in deep learning with PyTorch.

We use tools including large pre-trained models and model training dashboards to set up reproducible deep learning experiments and build machine learning models optimized for performance. There are several code examples throughout the training to help solidify the theoretical concepts that will be introduced. Models like Stable Diffusion, Llama 3, GPT, and BERT are highlighted as we uncover the training and optimization strategies to get the most of our models' performance, speed, and memory usage.
### Notebooks


#### 1. Introduction to Deep Learning

All data can be downloaded for the art classification example [here](https://drive.google.com/file/d/1jofGOHQ4PwZ50kpGuDqBeVXwDNcjPE6B/view?usp=sharing). Note it is about 6GB so it may take a bit.

- [**First steps with Deep Learning with MNIST**](notebooks/mnist.ipynb)
- [**RNNs and CNNs**](notebooks/rnn_and_cnn.ipynb)
- [**Working with pre-trained VGG-11 and BERT models**](notebooks/vgg_and_bert.ipynb)
- [**Fine-tuning BERT vs ChatGPT**](notebooks/BERT_vs_GPT_for_CLF.ipynb)
	- [Fine-tuning OpenAI](https://github.com/sinanuozdemir/quick-start-guide-to-llms/blob/main/notebooks/05_openai_app_review_fine_tuning.ipynb): the code to compare against BERT
- [**Fine-tuning GPT-2 to convert English to LaTEX**](notebooks/latex_gpt2.ipynb)
- [**Fine-tuning Llama 3 to be a chatbot**](https://colab.research.google.com/drive/1gN7jsUFQTPAj5uFrq06HcSLQSZzT7hZz?usp=sharing)

#### 2. Optimizing models 

- [**Production Optimization**](notebooks/deployment_and_optimization.ipynb)
- [**Quantizing Llama 3**](https://colab.research.google.com/drive/12RTnrcaXCeAqyGQNbWsrvcqKyOdr0NSm?usp=sharing)
- [**Testing different fine-tuning configurations**](https://colab.research.google.com/drive/1fdx2XlqfAjBoyiTktkRwa8SFaRF3Ch82?usp=sharing)
- [**Distilling BERT models**](https://github.com/sinanuozdemir/quick-start-guide-to-llms/blob/main/notebooks/11_distillation_example_2.ipynb)

#### 3. Going Further

- **[Intro to Multimodality](https://colab.research.google.com/drive/1zYSzDuYFa_cbRlti3scUjfmvradK8Sf4?usp=sharing)**: An introduction to multimodality with CLIP and SHAP-E + Diffusion

	- **[Whisper](https://colab.research.google.com/drive/1KxLWEEBtgix4zgP52pnxlIoJrZ8sHEYC?usp=sharing)**: An introduction to using Whisper for audio transcription

	- **[Llava](https://colab.research.google.com/drive/1IwNAz1Ee4YUSRNCU-SOsa7FS8Q2vmpoL?usp=sharing)**: Using an open source mult-turn multimodal engine
  
 	- **[CLIP-based Stock Image Search](https://colab.research.google.com/drive/1aUz0FKQDSAyXyhRyvkkRsSy7S30mpRJc?usp=sharing)**: Using CLIP to search through a library of images
  
  	- **[Dreambooth](https://colab.research.google.com/drive/1tQt1pE6l0MI79W8ZX0MMu0YVmF2I0GB3?usp=sharing)**: Fine-tuning a stable difusion model to make images of yours truly! Ever wonder what I look like blonde? Me neither but AI gave me some ideas of what it would look like.


- **Visual Q/A** - This case study requires you to [download the data from my Dropbox here](https://www.dropbox.com/scl/fo/w6iyfox8gnflvm7g10n47/AB47L7tNEl2Q8eyemZa2GMA?rlkey=v9s8bv6cmjukykpilzimswar0&st=fbulzw4e&dl=0). The code snippets should download them in code if that is easier! Our goal is to emulate the process done by [Llama 3.2-Vision-Instruct](https://colab.research.google.com/drive/1r6Nab2L7rYUBV5e8K8u8EFw98adJu5uh?usp=sharing): one of Meta's latest Llama models that can take in images.
	
	- Method 1: BERT + ViT -> GPT-2 (Fusion)

		- Constructing and Training our model: [Local](notebooks/constructing_a_vqa_system.ipynb) and notebook in [Colab](https://colab.research.google.com/drive/1zvbruS1DvFrVgXjNouSrrF9-PphKLWWl?usp=sharing)
		- Using our VQA system: [Local](notebooks/using_our_vqa.ipynb) notebook and [Colab](https://colab.research.google.com/drive/16GOBndQuIBO-UfXdpPte-PXaZS2nsW1H?usp=sharing)
	
		- Method 2: BERT + ViT -> GPT-2 (Fusion)
			- [Train the VQA Model](https://colab.research.google.com/drive/1DSh8_yfubuu5xPVM2BQ-I_eH5rrxLKZU?usp=sharing) and [use it here](https://colab.research.google.com/drive/1AWAk7NTvgTbjktUNB6bmS6T37bgTzRgt?usp=sharing)

#### How to Use the Image Recognition Flask App

`app.py` is a Flask app that uses a VGG16 model to classify the art style of an uploaded image. The app currently supports 10 different art styles:

- Abstract Expressionism
- Art Nouveau (Modern)
- Baroque
- Expressionism
- Impressionism
- Northern Renaissance
- Post-Impressionism
- Realism
- Romanticism
- Symbolism

Start the Flask app:
`python app.py`

This should start the Flask app and make it available at `http://localhost:5000`.

#### How to Use the App

To classify an image, you can use a cURL request in the following format:


```curl -X POST -F 'image=@/path/to/your/image.jpg' http://localhost:5000/predict```

Replace `/path/to/your/image.jpg` with the path to your own image. The response will be in JSON format and will contain the predicted art style and associated confidence scores, as shown below:

```
e.g.
curl -X POST -F \
  'image=@images/Venus_and_Adonis_by_Peter_Paul_Rubens.jpg' \
  http://localhost:5000/predict

[
	["Northern_Renaissance",0.13392961025238037],
	["Realism",0.12794768810272217],
	["Romanticism",0.12592236697673798],
	["Post_Impressionism",0.11863630264997482],
	["Baroque",0.11325731128454208],
	["Symbolism",0.1120268702507019],
	["Expressionism",0.08971412479877472],
	["Impressionism",0.086906298995018],
	["Art_Nouveau_Modern",0.05910796299576759],
	["Abstract_Expressionism",0.03255145251750946]]
```

If there is an error with the request, such as no image being provided, the response will contain an error message instead:

```
{
	"error": "No image provided"
}
```


## Instructor

**Sinan Ozdemir** is the Founder and CTO of LoopGenius where he uses State of the art AI to help people create and run their businesses. Sinan is a former lecturer of Data Science at Johns Hopkins University and the author of multiple textbooks on data science and machine learning. Additionally, he is the founder of the recently acquired Kylie.ai, an enterprise-grade conversational AI platform with RPA capabilities. He holds a master’s degree in Pure Mathematics from Johns Hopkins University and is based in San Francisco, CA.

# For More

- CHeck out [Deep Learning Illustrated](https://www.amazon.com/dp/0135116694?ref_=cm_sw_r_ffobk_cp_ud_dp_T500T43FCOX9F12OYRFO&peakEvent=5&dealEvent=0&bestFormat=true): A best seller by Jon Krohn, it's a very visual introduction to deep learning
- [Deep Learning course: lecture slides and lab notebooks](https://m2dsupsdlclass.github.io/lectures-labs/): The course covers the basics of Deep Learning, with a focus on applications.
