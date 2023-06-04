CUDA_VISIBLE_DEVICES=0 python main_agqa_v2.py --checkpoint_dir=agqa \
	--dataset=agqa \
	--mc=0 \
	--bnum=5 \
	--epochs=300 \
	--lr=0.00003 \
	--qmax_words=30 \
	--amax_words=38 \
	--max_feats=32 \
	--batch_size=128 \
	--batch_size_val=128 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--n_layers=2 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--feature_dim=512 \
	--dropout=0.3 \
	--seed=100 \
	--freq_display=150 \
	--save_dir='../data/save_models/agqa/mist_agqa_v2_fulldata/'


