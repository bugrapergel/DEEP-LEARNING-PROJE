# ══════════════════════════════════════════════════════════════════════════════
#  main.py  —  Uygulama Giriş Noktası
#  Çalıştırmak için:  python main.py
# ══════════════════════════════════════════════════════════════════════════════

from dashboard import Dashboard


def main():
    app = Dashboard()
    app.mainloop()


if __name__ == "__main__":
    main()