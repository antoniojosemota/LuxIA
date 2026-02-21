# LuxIA - Sistema de Controle de Iluminação Inteligente com TinyML

![LuxIA](https://img.shields.io/badge/LuxIA-TinyML-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![ESP32](https://img.shields.io/badge/ESP32-Compatible-green)

Sistema de controle automático de iluminação baseado em aprendizado de máquina, otimizado para dispositivos embarcados (TinyML). O modelo classifica a intensidade luminosa em três categorias e toma decisões automatizadas para controle de luzes.

## 📋 Visão Geral

O LuxIA utiliza uma rede neural artificial para classificar a luminosidade ambiente em tempo real, permitindo o controle inteligente de sistemas de iluminação. O modelo é treinado para reconhecer três estados distintos e executar ações correspondentes.

### Classes e Ações

| Classe | Condição | Ação |
|--------|----------|------|
| 0 | Escuro | Ligar luz |
| 1 | Claro | Manter |
| 2 | Muito Claro | Reduzir luz (desligar) |

## 🚀 Começando

### Pré-requisitos

- Python 3.7+
- Git
- TensorFlow 2.x
- ESP32 (para implementação embarcada)

### Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/antoniojosemota/LuxIA.git
cd LuxIA
```

### 📦 Instalação das Dependências

Execute o comando abaixo no terminal para instalar as bibliotecas necessárias:

```bash
pip install pandas numpy scikit-learn tensorflow
```

## 📊 Conjunto de Dados
O dataset (luxdata.csv) contém medições de intensidade luminosa em Lux, mapeadas para as três classes de ação. Os dados são pré-processados utilizando:

One-Hot Encoding: Conversão das classes em vetores binários

- Classe 0: [1, 0, 0]

- Classe 1: [0, 1, 0]

- Classe 2: [0, 0, 1]

Normalização: Divisão dos valores de lux por 1000.0

## 🧠 Arquitetura do Modelo
O modelo é construído com TensorFlow/Keras e possui a seguinte arquitetura:

```python
- Camada de Entrada: 1 neurônio (valor de lux normalizado)
- Camada Oculta: 8 neurônios com ativação ReLU
- Camada de Saída: 3 neurônios com ativação Softmax
```
### Parâmetros de Treinamento
- Otimizador: Adam

- Função de Perda: categorical_crossentropy

- Early Stopping: Monitoramento da perda de validação (patience=10)

## 🔧 Treinamento e Otimização
Treinar o Modelo
Execute o script principal para treinar o modelo:

```bash
python luxIA.py
```
Este script irá:

1. Carregar e pré-processar os dados

2. Treinar a rede neural

3. Salvar o modelo treinado

4. Gerar versões otimizadas

### Otimização para TinyML
O modelo é otimizado para execução em microcontroladores através de quantização INT8:

- Conversão de pesos de float32 para inteiros de 8 bits

- Redução significativa do tamanho do modelo

- Aumento da velocidade de inferência no ESP32

### Conversão para ESP32
Após o treinamento, converta o modelo para formato compatível com ESP32:

```bash
# Converter modelo TFLite para arquivo header C++
xxd -i lux_model_int8.tflite > lux_model.h
```

## 📁 Estrutura do Projeto
```text
LuxIA/
├── luxIA.py              # Script principal de treinamento
├── luxdata.csv           # Dataset de luminosidade
├── lux_model.tflite      # Modelo convertido para TFLite
├── lux_model.h           # Modelo quantizado INT8
└── README.md             # Este arquivo
```
## 🎯 Aplicação no ESP32
O modelo quantizado pode ser integrado em projetos ESP32 para:

- Leitura de sensores: Coleta de dados de luminosidade via BH1750

- Inferência local: Classificação em tempo real sem necessidade de nuvem

- Atuação: Controle automático de led

## Exemplo de uso no ESP32

[Hello World ML](https://github.com/antoniojosemota/Hello-World-ML.git)


## 📄 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

#### ✨ Autor
António José Mota - GitHub
