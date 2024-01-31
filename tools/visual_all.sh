#!/bin/bash
baseline=fast_base_tt_512_init
fsnet=fast_base_fsnet_tt_512_finetune_mixnet
ema=fast_base_ema_tt_512_finetune_ic17mlt
fsnet_ema=fast_base_fsnet_ema_tt_512_finetune_mixnet

# python test.py config/fast/tt/$ori.py pretrained/fast_base_tt_512_finetune_ic17mlt.pth --ema --min-score 0.85
# python test.py config/fast/tt/$fsnet.py pretrained/tt/$fsnet/checkpoint_17ep.pth.tar --ema --min-score 0.85
# python test.py config/fast/tt/$ema.py pretrained/tt/$ema/checkpoint_40ep.pth.tar --ema --min-score 0.84
# python test.py config/fast/tt/$fsnet_ema.py pretrained/tt/$fsnet_ema/checkpoint_12ep.pth.tar --ema --min-score 0.83


# cd eval 
# sh eval_tt.sh ../../outputs/$ori/submit_tt/
# sh eval_tt.sh ../../outputs/$fsnet/submit_tt/
# sh eval_tt.sh ../../outputs/$ema/submit_tt/
# sh eval_tt.sh ../../outputs/$fsnet_ema/submit_tt/
# cd ..

python visualize.py --dataset tt --config gt
# python visualize.py --dataset tt --config $ori
# python visualize.py --dataset tt --config $fsnet
# python visualize.py --dataset tt --config $ema
# python visualize.py --dataset tt --config $fsnet_ema

python tools/concatpic.py --pic-dir ori gt $baseline $ema $fsnet $fsnet_ema --dataset tt --title ori gt baseline ema fsnet fsnet_ema
