#!/bin/bash
baseline=fast_base_ctw_512_finetune_ic17mlt
ema=fast_base_ema_ctw_512_finetune_ic17mlt
fsnet=fast_base_fsnet_ctw_512_finetune_mixnet
fsnet_ema=fast_base_fsnet_ema_ctw_512_finetune_mixnet

python test.py config/fast/ctw/$baseline.py pretrained/fast_base_ctw_512_finetune_ic17mlt.pth --ema
python test.py config/fast/ctw/$ema.py pretrained/ctw/$ema/checkpoint_8ep.pth.tar --ema --min-score 0.86
python test.py config/fast/ctw/$fsnet.py pretrained/ctw/$fsnet/checkpoint_3ep.pth.tar --ema --min-score 0.80
python test.py config/fast/ctw/$fsnet_ema.py pretrained/ctw/$fsnet_ema/checkpoint_4ep.pth.tar --ema --min-score 0.83

cd eval 
sh eval_ctw.sh ../../outputs/$baseline/submit_ctw/
sh eval_ctw.sh ../../outputs/$ema/submit_ctw/
sh eval_ctw.sh ../../outputs/$fsnet/submit_ctw/
sh eval_ctw.sh ../../outputs/$fsnet_ema/submit_ctw/
cd ..

python visualize.py --dataset ctw --config gt
python visualize.py --dataset ctw --config $baseline
python visualize.py --dataset ctw --config $ema
python visualize.py --dataset ctw --config $fsnet
python visualize.py --dataset ctw --config $fsnet_ema

python tools/concatpic.py --pic-dir ori gt $baseline $ema $fsnet $fsnet_ema --dataset ctw --title ori gt baseline ema fsnet fsnet_ema
