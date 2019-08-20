from engine import translate
from vocab import prepareData, Lang
import torch 
from seq2seq import Encoder, Decoder
from vars import *

def load_model(path='files/model_De', device=device):
	checkpoint = torch.load(path, map_location=device)

	in_lang, out_lang, pairs = prepareData('En', 'De')
	in_lang  = checkpoint['in_lang_class']
	out_lang = checkpoint['out_lang_class']

	hidden_size = checkpoint['hidden_size']

	encoder = Encoder(in_lang.n_words, hidden_size).to(device)
	decoder = Decoder(hidden_size, out_lang.n_words, dropout_p=0.1).to(device)

	encoder.load_state_dict(checkpoint['encoder_state_dict'])
	decoder.load_state_dict(checkpoint['decoder_state_dict'])

	return encoder, decoder, in_lang, out_lang
# for testing purpose
if __name__ == '__main__':
	encoder, decoder, in_lang, out_lang = load_model()
	ans = translate('Ich bin gut.', encoder, decoder, in_lang, out_lang)
	print(ans)