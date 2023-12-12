CUDA_VISIBLE_DEVICES=6,7 python main_star.py --checkpoint_dir=star \
	--dataset=star \
	--mc=4 \
	--bnum=5 \
	--epochs=30 \
	--lr=0.00005 \
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
	--seed=300 \
	--freq_display=30 \
	--save_dir='../data/save_models/star/mist_rs_2_topk6_topj12_nlayer2_seed300_no_pos_8clips-lr5e-5_test2/' \

	
