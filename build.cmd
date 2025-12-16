cd D:\JasielC\Face\final_exe
pyinstaller --onefile --noconsole  --exclude-module=torch.distributed  --exclude-module=torch.distributed.algorithms  --exclude-module=torch.distributed.elastic --add-data="mouth_nano.pt;."  --add-data="tooth_nano.pt;."  --add-data="idle.png;." --add-data="smile.png;."   --add-data="promo.png;." tooth_recolour.py
