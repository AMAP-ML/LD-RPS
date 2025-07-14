HQ=results/enhancement;
GT=data/LOLv1/target;
python img_qua_ass/inference_iqa.py -m PSNR -i $HQ -r $GT;
python img_qua_ass/inference_iqa.py -m SSIM -i $HQ -r $GT;
python img_qua_ass/inference_iqa.py -m LPIPS -i $HQ -r $GT;
python img_qua_ass/inference_iqa.py -m PI -i $HQ -r $GT;
python img_qua_ass/inference_iqa.py -m NIQE -i $HQ -r $GT;

