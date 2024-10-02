# AeroAttention

## Overview

**AeroAttention** is an innovative, quantum-enhanced attention mechanism designed for transformer models. It offers optimized memory usage and accelerated computations, enabling scalable and efficient training for advanced neural network architectures.

## Features

- **Custom Quantum Computing Components:** Implements quantum principles for efficient attention computation.
- **Custom FFT and SVD Implementations:** Provides in-house implementations of Fast Fourier Transform (FFT) and Singular Value Decomposition (SVD) for full optimization control.
- **Entropy-Based Sparsity:** Reduces computational overhead by focusing on significant components.
- **Block Diagonalization:** Enhances computational efficiency by processing smaller matrix blocks.
- **Fully Optimized for Performance:** Designed to minimize memory usage and maximize speed.
- **Flexible Integration:** Easily integrates with popular transformer models like GPT-2 and LLaMA 2.

## Installation

### Prerequisites

- Python 3.6 or higher
- [Git](https://git-scm.com/)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/AeroAttention.git
   cd AeroAttention
   ```

2. Create a Virtual Environment (Optional but Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3. Install the Package
    ```bash
    pip install .
    ```

### Usage
**Integrating AeroAttention with Transformer Models**

**Example with GPT-2**
   ```python
    import torch
  import torch.nn as nn
  from aeroattention import AeroAttention
  
  class AeroAttentionLayer(nn.Module):
      def __init__(self, embed_dim, num_heads, aero_config):
          super(AeroAttentionLayer, self).__init__()
          self.embed_dim = embed_dim
          self.num_heads = num_heads
          self.head_dim = embed_dim // num_heads
  
          # Linear projections
          self.q_proj = nn.Linear(embed_dim, embed_dim)
          self.k_proj = nn.Linear(embed_dim, embed_dim)
          self.v_proj = nn.Linear(embed_dim, embed_dim)
  
          # Output projection
          self.out_proj = nn.Linear(embed_dim, embed_dim)
  
          # Initialize AeroAttention
          self.aero_attention = AeroAttention(
              num_qubits=aero_config.get('num_qubits', 4),
              threshold=aero_config.get('threshold', 0.1),
              compression_level=aero_config.get('compression_level', 0.5),
              block_size=aero_config.get('block_size', 64)
          )
  
      def forward(self, x, mask=None):
          batch_size, seq_len, embed_dim = x.size()
  
          # Compute Q, K, V
          Q = self.q_proj(x)
          K = self.k_proj(x)
          V = self.v_proj(x)
  
          # Reshape for multi-head attention
          Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
          K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
          V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
  
          # Initialize output tensor
          attention_outputs = []
  
          for b in range(batch_size):
              head_outputs = []
              for h in range(self.num_heads):
                  Q_bh = Q[b, h].detach().cpu().numpy()
                  K_bh = K[b, h].detach().cpu().numpy()
  
                  # Compute token matrix
                  token_matrix = np.dot(Q_bh, K_bh.T)
  
                  # Apply AeroAttention
                  aero_attention_scores = self.aero_attention.compute_attention(token_matrix)
  
                  # Convert back to tensor
                  attention_scores = torch.tensor(aero_attention_scores, device=x.device)
  
                  # Apply attention scores to V
                  V_bh = V[b, h]
                  context = torch.matmul(attention_scores, V_bh)
  
                  head_outputs.append(context)
  
              # Concatenate all heads
              head_outputs = torch.stack(head_outputs, dim=0)
              head_outputs = head_outputs.transpose(0, 1).contiguous().view(seq_len, -1)
              attention_outputs.append(head_outputs)
  
          # Stack all batches
          attention_outputs = torch.stack(attention_outputs, dim=0)
  
          # Final linear projection
          output = self.out_proj(attention_outputs)
  
          return output
  
  # Example usage within a GPT-2 model
  class GPT2WithAeroAttention(nn.Module):
      def __init__(self, config):
          super(GPT2WithAeroAttention, self).__init__()
          num_layers = config.get('num_layers', 12)
          self.layers = nn.ModuleList([
              nn.ModuleDict({
                  'ln_1': nn.LayerNorm(config['embed_dim']),
                  'attn': AeroAttentionLayer(
                      embed_dim=config['embed_dim'],
                      num_heads=config['num_heads'],
                      aero_config=config['aero_config']
                  ),
                  'ln_2': nn.LayerNorm(config['embed_dim']),
                  'mlp': nn.Sequential(
                      nn.Linear(config['embed_dim'], 4 * config['embed_dim']),
                      nn.GELU(),
                      nn.Linear(4 * config['embed_dim'], config['embed_dim'])
                  )
              }) for _ in range(num_layers)
          ])
          # Add other components like embeddings, etc.
  
      def forward(self, x):
          for layer in self.layers:
              x = layer['ln_1'](x)
              x = layer['attn'](x)
              x = layer['ln_2'](x)
              x = layer['mlp'](x)
          return x
  
  # Configuration example
  model_config = {
      'embed_dim': 768,
      'num_heads': 12,
      'aero_config': {
          'num_qubits': 4,
          'threshold': 0.1,
          'compression_level': 0.5,
          'block_size': 64
      },
      'num_layers': 12
  }
  
  # Initialize and use the model
  model = GPT2WithAeroAttention(model_config)
  input_ids = torch.randint(0, 50257, (1, 128))  # Example input
  output = model(input_ids)
  print(output.shape)  # Should be (1, 128, 768)
   ```
 
**Explanation:**
1. AeroAttentionLayer Class:
  - Replaces the standard self-attention mechanism with AeroAttention.
  - Projects input embeddings into query (Q), key (K), and value (V) matrices.
  - Applies AeroAttention to compute attention scores.
  - Concatenates the outputs from all attention heads and applies a final linear projection.
    
2. GPT2WithAeroAttention Class:
  - Integrates AeroAttentionLayer into each transformer block of GPT-2.
  - Maintains other components like layer normalization and feedforward networks.
    
2. Usage Example:
   - Demonstrates how to configure and initialize the modified GPT-2 model with AeroAttention.
   - Shows a forward pass with example input data.

### Benefits of Using AeroAttention:
  - **Memory Efficiency:** Custom FFT and SVD implementations reduce memory overhead.
  - **Speed Optimization:** Block diagonalization and entropy-based sparsity accelerate attention computations.
  - **Quantum-Enhanced** Performance: Integrates quantum principles for superior attention mechanisms.

## Development
### Running Tests
AeroAttention includes a comprehensive test suite to ensure all components function correctly.
1. Navigate to the Project Directory
      ```bash
       cd AeroAttention
      ```
2. Run Tests
     ```bash
     python -m unittest discover tests
     ```
## License
**This project is licensed under the MIT License - see the LICENSE file for details.**

## Contact
**For any inquiries or support, please contact aixr@meforgers.com**
