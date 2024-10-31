class DST_block:
    def __init__(self, Ut=None, Bt=None, At=0, Dt=None, DST_history=None):
        self.Ut = Ut if Ut is not None else []
        self.Bt = Bt if Bt is not None else []
        self.At = At
        self.Dt = Dt if Dt is not None else []
        self.DST_history = DST_history if DST_history is not None else []

    def update(self, *, Ut=None, Bt=None, At=None, Dt=None, DST_history=None):
        if Ut is not None:
            self.Ut = Ut
        if Bt is not None:
            self.Bt = Bt
        if At is not None:
            self.At = At
        if Dt is not None:
            self.Dt = Dt
        if DST_history is not None:
            self.DST_history = DST_history
    
    def __str__(self): 
        return (f"Ut: {self.Ut}, "
                f"Bt: {self.Bt}, "
                f"At: {self.At}, "
                f"Dt: {self.Dt}, "
                f"DST_history: {self.DST_history}")

# Khởi tạo đối tượng DST_block
dst = DST_block()
dst1 = DST_block()

# Cập nhật giá trị At với từ khóa
dst.update(At=3)

# Cập nhật giá trị Bt với từ khóa
dst.update(Bt=[1, 2, 3, 4])
# Cập nhật giá trị At với từ khóa
dst1.update(At=4)

# Cập nhật giá trị Bt với từ khóa
dst1.update(Bt=[5, 6, 7, 8])

dst.update(DST_history=dst1)

print(dst)
