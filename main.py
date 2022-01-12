import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR

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
