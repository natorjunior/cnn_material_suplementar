import streamlit as st
st.sidebar.title('CNN - (MATERIAL SUPLEMENTAR)')
paginas = ['LISTANDO ARQUIVOS','LENDO UMA IMAGEM', 'LENDO UM CONJUNTO DE IMAGENS', 'CRIANDO UMA CNN']

pages = st.sidebar.radio('selecione:',paginas)
if pages == 'LISTANDO ARQUIVOS':
    radios_ = st.radio('selecione:',['listagem simples','caminho completo'],horizontal=True)
    if radios_ == 'listagem simples':
        st.markdown('''
        Para listar arquivos em um diretório usando o módulo os do Python, você pode utilizar a função **os.listdir()**. Essa função retorna uma lista contendo os nomes dos arquivos e diretórios no caminho especificado. Aqui está um exemplo básico de como usá-la:
        ''')
        st.code('''
        import os

        # Especificar o diretório que você quer listar
        diretorio = '/caminho/para/o/diretorio'

        # Listar arquivos e diretórios
        arquivos = os.listdir(diretorio)

        # Exibir os nomes dos arquivos
        for arquivo in arquivos:
            print(arquivo)

        ''')
    elif radios_ == 'caminho completo':
        st.markdown('''Para obter o caminho completo de cada arquivo em um diretório, você pode combinar a função os.listdir() com **os.path.join()**. Aqui está como você pode fazer isso:''')
        st.code('''import os
# Especificar o diretório
diretorio = '/caminho/para/o/diretorio'

# Listar arquivos e diretórios
arquivos = os.listdir(diretorio)

# Exibir o caminho completo de cada arquivo
for arquivo in arquivos:
    caminho_completo = os.path.join(diretorio, arquivo)
    print(caminho_completo)
        ''')
elif pages == 'LENDO UMA IMAGEM':
    st.code('''from PIL import Image
import numpy as np
# Caminho para a imagem
caminho_da_imagem = '/caminho/para/sua/imagem.jpg'
# Ler a imagem
imagem = Image.open(caminho_da_imagem)
# Converter a imagem para um array NumPy
imagem_array = np.array(imagem)
# Exibir o array da imagem
print(imagem_array)
# Mostrar a imagem
imagem.show()
''')
elif pages == paginas[2]:
    st.code('''import os
from PIL import Image

# Especificar o diretório das imagens
diretorio_imagens = '/caminho/para/o/diretorio/das/imagens'

# Lista para armazenar as imagens
imagens = []

# Ler cada imagem no diretório
for arquivo in os.listdir(diretorio_imagens):
    if arquivo.endswith(".jpg") or arquivo.endswith(".png"): # Adicione outros formatos se necessário
        caminho_imagem = os.path.join(diretorio_imagens, arquivo)
        imagem = Image.open(caminho_imagem)
        imagens.append(imagem)

# Agora, 'imagens' contém todas as imagens do diretório
# Você pode processá-las conforme necessário
''')
elif pages == paginas[3]:
    st.code('''import numpy as np
import os
from tensorflow.keras.applications import VGG19
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.utils import to_categorical
import numpy as np
num_classes = 2
#VGG19 pré-treinado
#VGG19 sem as camadas totalmente conectadas
base_model = VGG19(weights='imagenet', include_top=False)

#camadas totalmente conectadas para a nova tarefa de classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

#Novo modelo combinando a base pré-treinada com as camadas personalizadas
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Tranformando os rótulos em codificação one-hot
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# Treinando o modelo
model.fit(x_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(x_test, y_test_one_hot))

# Medindo o desempenho do modelo nos dados de teste
accuracy = model.evaluate(x_test, y_test_one_hot)[1]
print(f'Acurácia nos dados de teste: {accuracy * 100:.2f}%')''')