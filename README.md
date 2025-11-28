# Taksi 6x6 Reinforcement Learning Projesi

Bu proje, Taksi probleminin 6x6 özel alan için tasarlanmış bir Reinforcement Learning (Q-Learning)uygulamasıdır.  
Taksinin yolcuyu aldığı doğru konumdan alıp hedefe ulaştırması ve duvar engelinden kaçınarak optimal bir öğrenme gerçekleştirmesidir.

## Proje Özeti

Bu projede ortam:

- 6x6 grid
- 4 yolcu konumu (o, o, o, o)
- 3 adet duvar var ve taksi bu hücrelere giremez.
- Sürekli yolcu alma – bırakma davranışı devam etmektedir.
- Eğitim gerçekleştirildikten sonra çıktıyı; GIF çıktısı olarak veriyor.

## Environment

### Grid Boyutu
- 6x6 (toplam 36 hücre)

### Duvar Sayısı
- 3 adet kapalı hücre mevcuttur.
- Bu hücrelere taksi giriş yapamaz.
- Duvarlar gri kare olarak yapıldı.

### Yolcu Sayısı
- 4 farklı sabit konum
  - o (Red) → (0, 0)  
  - o (Green) → (0, 5)  
  - o (Yellow) → (5, 0)  
  - o (Blue) → (5, 5)  
- Yolcu başlangıçta 4 konumdan birinde veya takside olabilir.

### Aksiyonlar
| Kod | Aksiyon |
|-----|---------|
| 0 | Güney (South) |
| 1 | Kuzey (North) |
| 2 | Doğu (East) |
| 3 | Batı (West) |
| 4 | Pickup |
| 5 | Dropoff |

---

### Episode (Epoch) Sayısı
- 8000 episode

## Ödül Sistemi (Reward Function)

Ajanın davranışlarını yönlendirmek için kullanılan ödül sistemi:

| Durum | Ödül |
|-------|------|
 Normal her adım  | (-1)
 Başarılı dropoff (doğru hedefe indirme) | (+20)
 Yanlış dropoff | (-10)
 Yanlış pickup | (-10)

- Ödül sayesinde öğrenme süresi hızlanmıştır. 
- Duvarlara gitmemeyi öğrenir. 
- Doğru pickup/dropoff davranışını öğrenir. 

##Gif Süreci 
İlk olarak eğitim başlangıcında (taxi.gif) sonucuna ulaştıktan sonra eğitimi geliştirerek sonrasında sonuç olarak (taxi6x6_policy.gif) dosyası ortaya çıkarılmıştır.
