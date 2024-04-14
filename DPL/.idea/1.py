class AC:
    def cool_wind(self):
        """吹冷风"""
        pass

    def hot_wind(self):
        """吹热风"""
        pass

    def swing(self):
        """摆头"""
        pass


class MD(AC):
    def cool_wind(self):
        print('冷风模式启动')

    def hot_wind(self):
        print('热风模式启动')

    def swing(self):
        print('开启左右摆头')


class GREE(AC):
    def cool_wind(self):
        print('冷风模式启动')

    def hot_wind(self):
        print('热风模式启动')

    def swing(self):
        print('开启左右摆头')


def make_cool(ac: AC):
    """开启摆头"""
    ac.swing()
    # """开始吹冷风"""
    # ac.cool_wind()
    # """开始吹热风"""
    # ac.cool_wind()


md_ac = MD()
gl_ac = GREE()


make_cool(md_ac)
make_cool(gl_ac)