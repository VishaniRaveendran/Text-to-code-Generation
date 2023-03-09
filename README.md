# Text-to-code-Generation
This notebook demonstrates text-to-code generation, which is the process of automatically generating code from natural language descriptions of software requirements.

<a id="title_ID"></a>
<font color='#436EEF'><center>
<h2>Text-to-code Generation</h>
</center></font>

<p>
This notebook demonstrates text-to-code generation, which is the process of automatically generating code from natural language descriptions of software requirements.
</p>

<a id="section1"><font color='#436EEE'><h2>Part A – Application area review</h2></font></a>

The literature review was conducted before implementing the model. The literature review found that there are various AI-based techniques that have been used for the purpose of generating code from natural language text. The literature review discussed the use of deep neural network (DNN) architectures such as Average Stochastic Gradient Descent (ASGD), Weight-Dropped LSTM (AWD-LSTM), QuasiRecurrent Neural Networks (QRNNs), and Transformer.[(Cruz-Benito et al., 2021)](https://www.zotero.org/google-docs/?mSMDpL).

Transformer models, in particular, have shown to be a promising approach for code generation. The transformer models that have been applied to code generation include GPT-2, BERT, and ROBERT, which have reportedly achieved exceptional results in the field of natural language processing. Additionally, transformer models such as Seq2Seq, Seq2Action+MAML, Iyer-Simp+200 idioms, GPT-2, CodeGPT, and CodeGPT-adapted have been explored in the field of text-to-code generation. [(Lu et al., 2021)](https://www.zotero.org/google-docs/?Dd6HhN). Furthermore, the CodeT5 pre-trained encoder-decoder transformer model significantly outperforms all previous work. [(Wang et al., 2021)](https://www.zotero.org/google-docs/?uKwuhB)

The ability of transformer models to handle long-term dependencies and their strong performance on natural language processing tasks were highlighted as the main reason for their potential effectiveness in code generation. Overall, it appears that transformer models are a promising approach for code generation.

<a id="section2"><font color='#436EEE'><h2>Part B – Compare and evaluate AI techniques</h2></font></a>

In this section, the focus is on comparing and evaluating various AI techniques such as Long Short-Term Memory (LSTM), Bidirectional Encoder Representations from Transformers (BERT), and CodeT5. The strengths and weaknesses, advantages and disadvantages of each technique are discussed, as well as providing brief examples of how each technique can be applied to a specific problem. A comparison and evaluation of the type of input data required and the expected output from each technique are also presented. 

<table>
  <tr>
   <td><strong>Technique</strong>
   </td>
   <td><strong>Strengths</strong>
   </td>
   <td><strong>Weaknesses</strong>
   </td>
   <td><strong>Advantages</strong>
   </td>
   <td><strong>Disadvantages</strong>
   </td>
   <td><strong>Input Data</strong>
   </td>
   <td><strong>Expected Output</strong>
   </td>
   <td><strong>Example</strong>
   </td>
  </tr>
  <tr>
   <td>BERT
   </td>
   <p>
   <td>BERT can comprehend the context of the text input and produce code accordingly.</p>
<p>
BERT's performance for text-to-code generation can be enhanced by fine-tuning it for a specific task and dataset.
   </td>
   <td>To achieve good text-to-code generation results, BERT may require a large amount of training data.</p>
<p>
BERT may struggle to understand complex and technical code-related natural language input.</p>
<p>
BERT may require a large amount of computational power in order to process and generate the code.</p>
   </td>
   <td>The model's accuracy is excellent because it is frequently updated. This is possible through successful fine-tuning training.</p>
<p>
For task-specific models, BERT works well.
<p>
Metrics can be fine-tuned and put to use right away.
   </td>
   <td>Because it is large and there are numerous weights to update, training takes a while.</p>
<p>
It costs a lot. Due to its size, it requires more computation, which has a cost.</p>
<p>
The model is large because of the training structure and corpus.
   </td>
   <td>Natural language text
   </td>
   <td>Code in programming languages such as Python, Java, etc.
   </td>
   <td>Code Summarization</p>
<p>
Code Generation from Natural Language</p>
<p>
Code Translation</p>
<p>
Code completion</p>
   </td>
  </tr>
  <tr>
   <td>LSTM
   </td>
   <td>LSTM can handle sequential data, such as code or natural language text, which makes it well-suited for code generation from natural language input.</p>
   </td>
   <td>LSTM may require large amounts of training data to achieve good results in text-to-code generation.</p>
<p>
LSTM may require a high computational power to process and generate the code.</p>
   </td>
   <td>LSTM is better at handling long-term dependencies, which is important when generating code from natural language input, as it allows the model to understand the context of the input text.</p>
<p>
LSTMs are very efficient at modeling complex sequential data, which makes them well-suited for code generation from natural language input.</p>
   </td>
   <td>LSTM require more training data in order to learn effectively.</p>
<p>
LSTMs can be slow to train on large datasets.</p> 
<p>
LSTMs may struggle with understanding complex and technical natural language input related to code, this may lead to a lower accuracy in the generated code.</p>
<p>
LSTMs may require a high computational power to process and generate the code.</p>
   </td>
   <td>Natural language text
   </td>
   <td>Code in programming languages such as Python, Java, etc.
   </td>
   <td>Code Summarization</p>
<p>
Code Generation from Natural Language</p>
<p>
Code Translation</p>
<p>
Code completion</p>
   </td>
  </tr>
  <tr>
   <td>CodeT5
   </td>
   <td>CodeT5 is pre-trained on a dataset of source code and natural language comments, which makes it well-suited for code generation from natural language input.</p>
<p>
CodeT5 uses the T5 architecture, which is a powerful transformer-based neural network model that can handle a wide range of natural language processing tasks.</p>
   </td>
   <td>CodeT5 is a relatively new technique and has not been widely used or evaluated yet.</p>
   </td>
   <td>CodeT5 uses the T5 architecture, which is a powerful transformer-based neural network model that can handle a wide range of natural language processing tasks.</p>
   </td>
   <td>CodeT5 is a relatively new technique and has not been widely used or evaluated yet, so its performance may be untested in certain situations.</p>
<p>
CodeT5 may require large amounts of training data to achieve good results.</p>
   </td>
   <td>Natural language text	
   </td>
   <td>	Generated code</p>
   </td>
   <td> Code summarization</p>
<p>
Code translation</p>
<p>
Code completion</p>
   </td>
  </tr>
</table>




Based on literature review, it appears CodeT5 is considered as a state-of-the-art technique for text-to-code generation. CodeT5 is pre-trained on a large dataset of source code and natural language comments, which makes it well-suited for code generation from natural language input. Additionally, the T5 architecture on which CodeT5 is based has been shown to achieve strong performance on a wide range of natural language processing tasks. Fine-tuning this model on a specific task like text-to-code generation would likely lead to improved performance compared to other techniques such as BERT or LSTM.

<a id="section2"><font color='#436EEE'><h2>Part C – Implementation</h2></font></a>

The input required for the implementation of text-to-code generation is the input required is a dataset of natural language text and corresponding source code.  The dataset used in this implementation was sourced from the Google dataset '[Mostly Basic Python Problems – Google Research](https://research.google/resources/datasets/mostly-basic-python-problems/)' which contains 1,000 crowd-sourced Python programming problems designed for entry-level programmers. The dataset includes task descriptions, code solutions and automated test cases.

The data was preprocessed by removing any irrelevant or noisy information, special characters, and stopwords through text cleaning. The text was tokenized and subword encoded using BPE technique to convert the tokens into numerical representations that can be used as input for the model.

To ensure the model generalizes well, the dataset was split into training, validation, and testing sets. The training set was used to train the model and learn the mapping between input text and corresponding output code. The validation set was used to fine-tune the model's hyperparameters and the testing set was used to evaluate the performance of the trained model.

The following diagram presents a high-level diagram of the process.
<br>
<br>
<center><img src="High level diagram.png" alt="fp16" width="600"/></center>
<br>




