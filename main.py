from pydoc import doc
import cv2
import numpy as np
import glob
from sklearn.svm import SVC
from sklearn.svm import SVR

#ler o diretorio(caminho) das imagens
def readpath(filepath):
	filenames = []
	files_path = [f for f in glob.glob(filepath)]

	for filename in files_path:
		if 'segmentation' not in filename:
			filenames.append(filename)
		if len(filenames) == 10:
			break

	return filenames

#lendo as imagens
def read_files(imgs):
	readed_imgs = []
	for img in imgs:
		readed_imgs.append(cv2.imread(img))
	return readed_imgs

#alterando tamanho das imagens
def resize_img(imgs,size):
	new_imgs = []
	for img in imgs:
		new_imgs.append(cv2.resize(img,size))

	return new_imgs

	
'''
def reshape_imgs(imgs,size):
	new_imgs = []
	for img in imgs:
		new_imgs.append(cv2.reshape(img,size))
	return tuple(new_imgs)
'''


def call_funcions(document,size : tuple):
	document = readpath(document)
	document = read_files(document)
	document = resize_img(document,size)
	return document


#Apenas testes para renderizar a imagem
cnh_aberta = call_funcions('dataset/CNH_Aberta/*.jpg',(400,400))
cnh_frente = call_funcions('dataset/CNH_Frente/*.jpg',(400,400))
cnh_verso = call_funcions('dataset/CNH_Verso/*.jpg',(400,400))

cpf_frente = call_funcions('dataset/CPF_Frente/*.jpg',(400,400))
cpf_verso = call_funcions('dataset/CPF_Verso/*.jpg',(400,400))

rg_frente = call_funcions('dataset/RG_Frente/*.jpg',(400,400))
rg_verso = call_funcions('dataset/RG_Verso/*.jpg',(400,400))
rg_aberto = call_funcions('dataset/RG_Aberto/*.jpg',(400,400))

labels = [1,2,3,4,5,6,7,8]
documents = []

for a,b,c,d,e,f,g,h in zip(cnh_aberta,cnh_frente,cnh_verso,cpf_frente,cpf_verso,rg_frente,rg_verso,rg_aberto):
	documents.append({'x':a, 'y':labels[0]})
	documents.append({'x':b,'y':labels[1]})
	documents.append({'x':c,'y':labels[2]})
	documents.append({'x':d,'y':labels[3]})
	documents.append({'x':e,'y':labels[4]})
	documents.append({'x':f,'y':labels[5]})
	documents.append({'x':g,'y':labels[6]})
	documents.append({'x':h,'y':labels[7]})


'''
for p in documents:
	cv2.imshow("Test",p['x'])
	print(p['y'])
	cv2.waitKey(0)
'''

x_doc = []
y_doc = []

for document in documents:
	x_doc.append(document['x']) 
	y_doc.append(document['y'])

'''
for p in x_doc:
	cv2.imshow("Test",p)
	cv2.waitKey(0)
'''

x_doc = np.concatenate(x_doc,axis=0)

y_doc = np.array(y_doc)
y_doc = y_doc.reshape(-1)

'''
# documents
x_doc = np.concatenate((cnh_frente,cnh_verso,cnh_aberta,rg_frente,rg_verso,rg_aberto,cpf_frente,cpf_verso),axis=0)
y_doc = [1,2,3,4,5,6,7,8]

#documents
y_doc = np.array(y_doc)
y_doc = y_doc.reshape(-1)

x_doc = x_doc.reshape(len(y_doc),-1)

#document classifier
document_classifier = SVC(kernel='linear')

print(40 * '-')
print('Started train of SVC model')

#Train document with images and indexes
document_classifier.fit(x_doc,y_doc)


print('Finished train')
print(40 * '-')

#documents

prediction_d  = document_classifier.predict(rg_frente_test.reshape(1,-1))

score_d = document_classifier.score(x_doc,y_doc)

# Show prediction
print('Result: {}'.format(prediction_d))
# Show prediction score
print('Score of precision: {:.1f}%'.format(score_d * 100))

if prediction_d == 1:
	result = cnh_frente
elif prediction_d == 2:
	result = cnh_verso
elif prediction_d == 3:
	result = cnh_aberta
elif prediction_d == 4:
	result = rg_frente
elif prediction_d == 5:
	result = rg_verso
elif prediction_d == 6:
	result = rg_aberto
elif prediction_d == 7:
	result = cpf_frente
elif prediction_d == 8:
	result = cpf_verso

# Show image based on prediction
cv2.imshow("Result", result)
# Show the image tested
cv2.imshow("Test", rg_frente_test)
# Wait for key
cv2.waitKey(0)

print('---------------------------------------')
'''

'''
# Read all documents
cnh_aberta = cv2.imread('images/cnh_aberta.jpg')
cnh_frente = cv2.imread('images/cnh_frente.jpg')
cnh_verso = cv2.imread('images/cnh_verso.jpg')
rg_aberto = cv2.imread('images/rg_aberto.jpg')
rg_frente = cv2.imread('images/rg_verso.jpg')
rg_verso = cv2.imread('images/rg_verso.jpg')
cpf_frente = cv2.imread('images/cpf_frente.jpg')
cpf_verso = cv2.imread('images/cpf_verso.jpg')

rg_frente_test = cv2.imread('images/cpf_test.jpg')

cnh_aberta = cv2.resize(cnh_aberta,(200,200))
cnh_frente = cv2.resize(cnh_frente,(200,200))
cnh_verso = cv2.resize(cnh_verso,(200,200))
rg_aberto = cv2.resize(rg_aberto,(200,200))
rg_frente = cv2.resize(rg_frente,(200,200))
rg_verso = cv2.resize(rg_verso,(200,200))
cpf_frente = cv2.resize(cpf_frente,(200,200))
cpf_verso = cv2.resize(cpf_verso,(200,200))

rg_frente_test =cv2.resize(rg_frente_test,(200,200))
'''