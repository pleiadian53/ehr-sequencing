"""
Unit tests for LSTM baseline model.

Tests:
- VisitEncoder with different aggregation strategies
- LSTMBaseline forward pass
- Model prediction
- create_lstm_baseline factory function
"""

import pytest
import torch
from ehrsequencing.models import LSTMBaseline, VisitEncoder, create_lstm_baseline


class TestVisitEncoder:
    """Test VisitEncoder."""
    
    def test_mean_aggregation(self):
        """Test mean aggregation."""
        encoder = VisitEncoder(embedding_dim=128, aggregation='mean')
        
        # Create mock visit embeddings
        batch_size, num_codes, embed_dim = 4, 10, 128
        visit_embeddings = torch.randn(batch_size, num_codes, embed_dim)
        visit_mask = torch.ones(batch_size, num_codes)
        
        # Encode
        visit_vector = encoder(visit_embeddings, visit_mask)
        
        assert visit_vector.shape == (batch_size, embed_dim)
    
    def test_attention_aggregation(self):
        """Test attention aggregation."""
        encoder = VisitEncoder(embedding_dim=128, aggregation='attention')
        
        batch_size, num_codes, embed_dim = 4, 10, 128
        visit_embeddings = torch.randn(batch_size, num_codes, embed_dim)
        visit_mask = torch.ones(batch_size, num_codes)
        
        visit_vector = encoder(visit_embeddings, visit_mask)
        
        assert visit_vector.shape == (batch_size, embed_dim)
    
    def test_masked_aggregation(self):
        """Test that padding is properly masked."""
        encoder = VisitEncoder(embedding_dim=128, aggregation='mean')
        
        batch_size, num_codes, embed_dim = 2, 10, 128
        visit_embeddings = torch.randn(batch_size, num_codes, embed_dim)
        
        # Create mask: first sequence has 5 real codes, second has 8
        visit_mask = torch.zeros(batch_size, num_codes)
        visit_mask[0, :5] = 1
        visit_mask[1, :8] = 1
        
        visit_vector = encoder(visit_embeddings, visit_mask)
        
        assert visit_vector.shape == (batch_size, embed_dim)
        # Verify no NaN values
        assert not torch.isnan(visit_vector).any()


class TestLSTMBaseline:
    """Test LSTMBaseline model."""
    
    @pytest.fixture
    def model_config(self):
        """Default model configuration."""
        return {
            'vocab_size': 1000,
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'output_dim': 1,
            'task': 'binary_classification'
        }
    
    def test_model_initialization(self, model_config):
        """Test model can be initialized."""
        model = LSTMBaseline(**model_config)
        
        assert model.vocab_size == 1000
        assert model.embedding_dim == 64
        assert model.hidden_dim == 128
        assert model.num_layers == 2
    
    def test_forward_pass(self, model_config):
        """Test forward pass with mock data."""
        model = LSTMBaseline(**model_config)
        
        # Create mock input
        batch_size, num_visits, max_codes = 4, 10, 20
        visit_codes = torch.randint(0, 1000, (batch_size, num_visits, max_codes))
        visit_mask = torch.ones(batch_size, num_visits, max_codes, dtype=torch.bool)
        sequence_mask = torch.ones(batch_size, num_visits, dtype=torch.bool)
        
        # Forward pass
        output = model(visit_codes, visit_mask, sequence_mask)
        
        assert 'logits' in output
        assert 'predictions' in output
        assert output['logits'].shape == (batch_size, 1)
        assert output['predictions'].shape == (batch_size, 1)
        
        # Check predictions are in valid range for binary classification
        assert (output['predictions'] >= 0).all()
        assert (output['predictions'] <= 1).all()
    
    def test_variable_length_sequences(self, model_config):
        """Test model handles variable-length sequences."""
        model = LSTMBaseline(**model_config)
        
        batch_size, num_visits, max_codes = 4, 10, 20
        visit_codes = torch.randint(0, 1000, (batch_size, num_visits, max_codes))
        
        # Create variable-length masks
        visit_mask = torch.zeros(batch_size, num_visits, max_codes, dtype=torch.bool)
        sequence_mask = torch.zeros(batch_size, num_visits, dtype=torch.bool)
        
        # First sequence: 5 visits, varying codes per visit
        sequence_mask[0, :5] = 1
        visit_mask[0, 0, :10] = 1
        visit_mask[0, 1, :15] = 1
        visit_mask[0, 2, :8] = 1
        visit_mask[0, 3, :12] = 1
        visit_mask[0, 4, :5] = 1
        
        # Second sequence: 8 visits
        sequence_mask[1, :8] = 1
        for i in range(8):
            visit_mask[1, i, :10] = 1
        
        # Third sequence: 3 visits
        sequence_mask[2, :3] = 1
        for i in range(3):
            visit_mask[2, i, :5] = 1
        
        # Fourth sequence: 6 visits
        sequence_mask[3, :6] = 1
        for i in range(6):
            visit_mask[3, i, :8] = 1
        
        # Forward pass
        output = model(visit_codes, visit_mask, sequence_mask)
        
        assert output['logits'].shape == (batch_size, 1)
        assert not torch.isnan(output['logits']).any()
    
    def test_bidirectional_lstm(self, model_config):
        """Test bidirectional LSTM."""
        model_config['bidirectional'] = True
        model = LSTMBaseline(**model_config)
        
        batch_size, num_visits, max_codes = 4, 10, 20
        visit_codes = torch.randint(0, 1000, (batch_size, num_visits, max_codes))
        
        output = model(visit_codes)
        
        assert output['logits'].shape == (batch_size, 1)
    
    def test_return_hidden_states(self, model_config):
        """Test returning hidden states."""
        model = LSTMBaseline(**model_config)
        
        batch_size, num_visits, max_codes = 4, 10, 20
        visit_codes = torch.randint(0, 1000, (batch_size, num_visits, max_codes))
        
        output = model(visit_codes, return_hidden=True)
        
        assert 'hidden_states' in output
        assert output['hidden_states'].shape == (batch_size, num_visits, model_config['hidden_dim'])
    
    def test_predict_method(self, model_config):
        """Test predict convenience method."""
        model = LSTMBaseline(**model_config)
        model.eval()
        
        batch_size, num_visits, max_codes = 4, 10, 20
        visit_codes = torch.randint(0, 1000, (batch_size, num_visits, max_codes))
        
        predictions = model.predict(visit_codes)
        
        assert predictions.shape == (batch_size, 1)
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()
    
    def test_get_embeddings(self, model_config):
        """Test getting code embeddings."""
        model = LSTMBaseline(**model_config)
        
        codes = torch.tensor([[1, 2, 3], [4, 5, 6]])
        embeddings = model.get_embeddings(codes)
        
        assert embeddings.shape == (2, 3, model_config['embedding_dim'])


class TestCreateLSTMBaseline:
    """Test create_lstm_baseline factory function."""
    
    def test_small_model(self):
        """Test creating small model."""
        model = create_lstm_baseline(
            vocab_size=1000,
            task='binary_classification',
            model_size='small'
        )
        
        assert model.embedding_dim == 128
        assert model.hidden_dim == 256
        assert model.num_layers == 1
    
    def test_medium_model(self):
        """Test creating medium model."""
        model = create_lstm_baseline(
            vocab_size=1000,
            task='binary_classification',
            model_size='medium'
        )
        
        assert model.embedding_dim == 256
        assert model.hidden_dim == 512
        assert model.num_layers == 2
    
    def test_large_model(self):
        """Test creating large model."""
        model = create_lstm_baseline(
            vocab_size=1000,
            task='binary_classification',
            model_size='large'
        )
        
        assert model.embedding_dim == 512
        assert model.hidden_dim == 1024
        assert model.num_layers == 3
    
    def test_multi_class_task(self):
        """Test creating model for multi-class classification."""
        model = create_lstm_baseline(
            vocab_size=1000,
            task='multi_class',
            output_dim=10,
            model_size='small'
        )
        
        assert model.output_dim == 10
        assert model.task == 'multi_class'
    
    def test_regression_task(self):
        """Test creating model for regression."""
        model = create_lstm_baseline(
            vocab_size=1000,
            task='regression',
            model_size='small'
        )
        
        assert model.output_dim == 1
        assert model.task == 'regression'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
