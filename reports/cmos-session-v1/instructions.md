# CMOS dinleme oturumu — yönerge (v1)

Kulaklık kullanın. Her denemede aynı numaralı iki kayıt var: `trial-NN-A.wav` ve `trial-NN-B.wav`.
`trials.csv` dosyasındaki `content` sütunu söylenen (veya bağlam) metnini gösterir.

Her deneme için iki soruyu yanıtlayın ve bir tabloya yazın:

1. **Karşılaştırma (−3 … +3):** Hangisi bu içeriğin devamı olarak daha doğal ve aynı kişinin
   sesi gibi geliyor? −3 = A açıkça daha iyi, 0 = fark yok, +3 = B açıkça daha iyi.
2. **Aynı kişi mi? (E/H):** İki kayıt da aynı kişiye mi ait?

Kurallar:
- Kayıtları en fazla iki kez dinleyin.
- İçerik farklı olan denemelerde (type=anchored) yalnızca ses kimliğine ve doğallığa odaklanın.
- Hangi sistemin hangisi olduğu bilinmiyor; `key.json` dosyasını oturum bitmeden AÇMAYIN.

Sonuçları `results-<isim>.csv` olarak şu kolonlarla kaydedin: `trial, cmos, same_person`.
