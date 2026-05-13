import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.pointpillars import PointPillars
from dataset.kitti_dataset import KITTIDataset
import time

# Hiperparametreler (Makalede (Raporda) Belirtilen Değerler)
LEARNING_RATE = 2e-4
EPOCHS = 160
BATCH_SIZE = 4
PRETRAINED_WEIGHTS = 'weights/pointpillars_pretrained.pth'
FINETUNED_WEIGHTS = 'weights/pointpillars_finetuned_v5.pth'


def train_model():
    print("[BİLGİ] KITTI Veri Seti Yükleniyor...")
    train_dataset = KITTIDataset(root_path='KITTI/training', split='train', augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("[BİLGİ] PointPillars Mimarisi Başlatılıyor...")
    model = PointPillars(num_classes=3)  # Car, Pedestrian, Cyclist

    # Pre-trained (Önceden eğitilmiş) ağırlıkları yükle
    print(f"[BİLGİ] Önceden eğitilmiş ağırlıklar yükleniyor: {PRETRAINED_WEIGHTS}")
    model.load_state_dict(torch.load(PRETRAINED_WEIGHTS), strict=False)

    model.cuda()
    model.train()

    # Raporda bahsettiğimiz Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    print(f"[BİLGİ] Fine-Tuning İşlemi Başlıyor... Toplam Epoch: {EPOCHS}")

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, data_dict in enumerate(train_loader):
            # Verileri GPU'ya taşı
            for key, val in data_dict.items():
                if isinstance(val, torch.Tensor):
                    data_dict[key] = val.cuda()

            optimizer.zero_grad()

            # İleri Besleme (Forward Pass)
            loss, tb_dict, disp_dict = model(data_dict)

            # Geri Yayılım (Backward Pass)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        end_time = time.time()

        # Sadece her 10 epoch'ta bir log yazdır (Terminal temizliği için)
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] | Kayıp (Loss): {epoch_loss / len(train_loader):.4f} | "
                  f"Süre: {end_time - start_time:.2f} sn | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Fine-tune edilmiş yeni ağırlıkları kaydet
    print(f"[BAŞARILI] Fine-Tuning Tamamlandı! Yeni ağırlıklar kaydediliyor: {FINETUNED_WEIGHTS}")
    torch.save(model.state_dict(), FINETUNED_WEIGHTS)


if __name__ == '__main__':
    train_model()