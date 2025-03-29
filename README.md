<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
  <h1>CapitalX Credit Card Recommendation System</h1>

  <div class="overview">
    <p>A fine-tuned LLM system that recommends optimal credit cards for specific transactions based on rewards, APR, and other factors.</p>
  </div>

  <h2>📋 Overview</h2>
  <p>
    This system employs Parameter-Efficient Fine-Tuning (PEFT) to adapt foundation models for intelligent credit card recommendations, analyzing transaction details against multiple card options to maximize customer benefits.
  </p>

  <h2>🏗️ Architecture</h2>
  <div class="architecture">
    <ul>
      <li><strong>Base Model:</strong> <code>meta-llama/Meta-Llama-3-70B</code> (70B parameter language model)</li>
      <li><strong>Distributed Training:</strong> Modal cloud with DeepSpeed ZeRO-3 optimization</li>
      <li><strong>Memory Optimization:</strong> QLoRA (Quantized Low-Rank Adaptation) with 4-bit quantization</li>
      <li><strong>Hardware Requirements:</strong> 8x H100 GPUs for training, 1x H100 for inference</li>
    </ul>
  </div>

  <h2>⚙️ Technical Details</h2>

  <h3>Training Configuration</h3>
  <table>
    <tr>
      <th>Parameter</th>
      <th>Value</th>
    </tr>
    <tr>
      <td>Learning Rate</td>
      <td>5e-6</td>
    </tr>
    <tr>
      <td>Training Epochs</td>
      <td>2</td>
    </tr>
    <tr>
      <td>LoRA Rank (r)</td>
      <td>16</td>
    </tr>
    <tr>
      <td>LoRA Alpha</td>
      <td>32</td>
    </tr>
    <tr>
      <td>Batch Size Per Device</td>
      <td>2</td>
    </tr>
    <tr>
      <td>Gradient Accumulation Steps</td>
      <td>4</td>
    </tr>
  </table>

  <h3>Data Processing</h3>
  <p>
    Each training example follows a standardized instruction format:
  </p>

  <pre>You are a credit card recommendation assistant. Analyze the cards and transaction, then recommend the BEST card with a clear explanation.

Cards:
Freedom Flex: APR 15.0%, Credit Limit 5000, Rewards: base 1.0, dining 3.0%, groceries 5.0%
Sapphire Preferred: APR 18.0%, Credit Limit 10000, Rewards: base 1.0, travel 4.0%, dining 3.0%

Transaction:
Product: Dinner, Category: dining, Vendor: Restaurant, Price: 75.0

Output: </pre>

  <h2>🚀 Implementation</h2>

  <div class="prerequisites">
    <h4>Prerequisites</h4>
    <ul>
      <li>Hugging Face API token with access to required models</li>
      <li>Modal cloud account configured with GPU access</li>
      <li>Python 3.8+ with required dependencies</li>
    </ul>
  </div>

  <h3>Environment Setup</h3>

  <pre># Create .env file with Hugging Face token
echo "HUGGING_FACE_TOKEN=your_token_here" > .env

# Run the script
modal run capitalx.py</pre>

  <h3>Command-line Options</h3>

  <table>
    <tr>
      <th>Option</th>
      <th>Description</th>
    </tr>
    <tr>
      <td><code>--model_name</code></td>
      <td>Specify a different base model (default: meta-llama/Meta-Llama-3-70B)</td>
    </tr>
    <tr>
      <td><code>--test_auth</code></td>
      <td>Test Hugging Face authentication only</td>
    </tr>
    <tr>
      <td><code>--test_gpus</code></td>
      <td>Test GPU setup and compatibility only</td>
    </tr>
  </table>

  <h2>💡 Example Output</h2>

  <pre>{
  "test_case_1": "Best card: Freedom Flex. Explanation: Both cards offer the same 3% rewards on dining, but Freedom Flex has a lower APR (15.0% vs 18.0%), making it the better choice for this transaction.",
  "test_case_2": "Best card: Freedom Flex. Explanation: Freedom Flex offers 5% cash back on groceries compared to no specific grocery rewards with Sapphire Preferred, resulting in $6.00 in rewards versus just $1.20 with the base rate."
}</pre>

  <h2>🔍 System Workflow</h2>

  <div class="workflow">
    <ol>
      <li><strong>GPU Testing:</strong> Validates CUDA availability and multi-GPU setup</li>
      <li><strong>Data Preprocessing:</strong> Formats training data with instruction prefixes</li>
      <li><strong>Model Loading:</strong> Initializes the base model with 4-bit quantization</li>
      <li><strong>Fine-Tuning:</strong> Applies LoRA to adapt the model using DeepSpeed ZeRO-3</li>
      <li><strong>Evaluation:</strong> Tests the trained model on sample credit card recommendation scenarios</li>
      <li><strong>Result Reporting:</strong> Displays recommendation results for validation</li>
    </ol>
  </div>

  <h2>📝 License & Acknowledgements</h2>
  <p>
    This project uses the Meta-Llama-3-70B model from Meta under their model license. Please ensure you have proper access rights and comply with the model's usage terms.
  </p>

  <div class="footer">
    <p>For questions or contributions, please open an issue in the repository.</p>
  </div>
</body>
</html>
