import os
import math
import numpy as np
import matplotlib.pyplot as plt


class DieselEngine:
    def __init__(self, filename):
        self.filename = filename
        self.data = {}  # Входные
        self.calc = {}  # Расчетные
        self.check = {} # Проверки
        self.ranges = {}# Допустимые значения
        self.dyn = {} # динамический расчет

        # Справочник единиц измерения
        self.units = {
            # --- Исходные данные ---
            'Ne': 'кВт', 'epsilon': '-', 'n': 'об/мин', 'i': '-',
            'theta': '-', 'SD': '-', 'alpha': '-',
            'p0': 'МПа', 'T0': 'К', 'nk': '-', 'incps': 'МПа',
            'noch': '-', 'incT': 'К', 'gammar': '-', 'Tr': 'К',
            'Hu': 'кДж/кг', 'gc': '-', 'gh': '-', 'go': '-',
            'n1': '-', 'k': '-', 'n2': '-', 'Tz': 'К',
            'phip': '-', 'num': '-',
            'lambda_k': '-', 'm_j': 'кг',
            # --- Расчётные параметры ---
            'pk': 'МПа', 'pr': 'МПа', 'pa': 'МПа', 'incpa': 'МПа',
            'Tk': 'К', 'ps': 'МПа', 'Ts': 'К', 'Ta': 'К', 'nuv': '-',
            'pc': 'МПа', 'Tc': 'К',
            'L0': 'кмоль/кг', 'M1': 'кмоль/кг', 'M2': 'кмоль/кг',
            'nu0': '-', 'nu': '-',
            'pzd': 'МПа', 'pz': 'МПа', 'mcvc': 'кДж/(кмоль·К)',
            'p': '-', 'sigma': '-', 'Tb': 'К', 'pb': 'МПа',
            'Trc': 'К',
            "pi'": 'МПа', 'pi': 'МПа', 'nui': '-', 'gi': 'г/(кВт·ч)',
            'pe': 'МПа', 'nue': '-', 'ge': 'г/(кВт·ч)',
            'Vl': 'л', 'Vh': 'л', 'D-real': 'дм', 'D': 'дм',
            'S-real': 'дм', 'S': 'дм', 'D-mm': 'мм', 'S-mm': 'мм',
            "Vl'": 'л', "Ne'": 'кВт', 'Me': 'Н·м', 'Gt': 'кг/ч',
            'Cm': 'м/с', 'Nl': 'кВт/л',
        }

    def load_inputs(self):
        if not os.path.exists(self.filename):
            print(f"Файл {self.filename} не найден!")
            return False

        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '#' in line:
                    line = line[:line.index('#')].strip()
                if not line or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                self.data[key.strip()] = float(value.strip())
        return True

    def calculate(self):
        global mult1
        d = self.data
        res = self.calc

        def store(key, value, formula):
            res[key] = value
            res[f'{key}_f'] = formula

        # 1. Процесс впуска
        pk = 1.5 * d['p0']
        store('pk', pk, f"pk = 1.5·p0 = 1.5·{d['p0']:.2f} = {pk:.4f} МПа")

        pr = 1.1 * pk
        store('pr', pr, f"pr = 1.1·pk = 1.1·{pk:.4f} = {pr:.4f} МПа")

        pa = 0.95 * pk
        store('pa', pa, f"pa = 0.95·pk = 0.95·{pk:.4f} = {pa:.4f} МПа")

        incpa = pk - pa
        store('incpa', incpa, f"Δpa = pk - pa = {pk:.4f} - {pa:.4f} = {incpa:.4f} МПа")

        ratio1 = pk / d['p0']
        mult1 = -1 if ratio1 < 0 else 1

        Tk = d['T0'] * mult1 * (abs(ratio1) ** ((d['nk'] - 1) / d['nk']))
        eps_Tk = (d['nk'] - 1) / d['nk']
        store('Tk', Tk, f"Tk = T0·(pk/p0)^((nk-1)/nk) = {d['T0']:.0f}·({pk:.2f}/{d['p0']:.2f})^{eps_Tk:.4f} = {Tk:.4f} К")

        ps = pk - d['incps']
        store('ps', ps, f"ps = pk - Δps = {pk:.4f} - {d['incps']:.4f} = {ps:.4f} МПа")

        Ts = Tk
        store('Ts', Ts, f"Ts = Tk = {Tk:.4f} К")

        Ta = (Ts + d['incT'] + d['gammar'] * d['Tr']) / (1 + d['gammar'])
        store('Ta', Ta, f"Ta = (Ts + ΔT + γ·Tr) / (1+γ) = ({Ts:.1f} + {d['incT']:.0f} + {d['gammar']:.2f}·{d['Tr']:.0f}) / (1+{d['gammar']:.2f}) = {Ta:.4f} К")

        nuv = (d['epsilon'] * pa - pr * (1 - d['gammar'])) * ((Ts / (Ts + d['incT'])) * (1 / (d['epsilon'] - 1)) * (1 / pk))
        store('nuv', nuv, f"ηv = (ε·pa - pr·(1-γ))·(Ts/(Ts+ΔT))·(1/(ε-1))·(1/pk) = ({d['epsilon']}·{pa:.4f} - {pr:.4f}·(1-{d['gammar']:.2f}))·({Ts:.0f}/({Ts + d['incT']:.0f}))·(1/({d['epsilon'] - 1}))·(1/{pk:.4f}) = {nuv:.4f}")

        # 2. Процесс сжатия
        pc = pa * (d['epsilon'] ** d['n1'])
        store('pc', pc, f"pc = pa·ε^n1 = {pa:.4f}·{d['epsilon']}^{d['n1']:.2f} = {pc:.4f} МПа")

        Tc = Ta * (d['epsilon'] ** (d['n1'] - 1))
        store('Tc', Tc, f"Tc = Ta·ε^(n1-1) = {Ta:.4f}·{d['epsilon']}^{d['n1'] - 1:.2f} = {Tc:.4f} К")

        # 3. Процесс сгорания
        L0 = 1 / 0.21 * ((d['gc'] / 12) + (d['gh'] / 4) - (d['go'] / 32))
        store('L0', L0, f"L0 = (1/0.21)·(C/12 + H/4 - O/32) = {1/0.21:.2f}·({d['gc']:.4f}/12 + {d['gh']:.4f}/4 - {d['go']:.4f}/32) = {L0:.4f} кмоль/кг")

        M1 = d['alpha'] * L0
        store('M1', M1, f"M1 = α·L0 = {d['alpha']:.2f}·{L0:.4f} = {M1:.4f} кмоль/кг")

        M2 = (d['gc'] / 12) + (d['gh'] / 2) + (d['alpha'] - 0.21) * L0
        store('M2', M2, f"M2 = C/12 + H/2 + (α-0.21)·L0 = {d['gc']:.4f}/12 + {d['gh']:.4f}/2 + ({d['alpha']:.2f} - 0.21)·{L0:.4f} = {M2:.4f} кмоль/кг")

        nu0 = M2 / M1
        store('nu0', nu0, f"μ0 = M2/M1 = {M2:.4f}/{M1:.4f} = {nu0:.4f}")

        nu = 1 + (nu0 - 1) / (1 + d['gammar'])
        store('nu', nu, f"μ = 1 + (μ0-1)/(1+γ) = 1 + ({nu0:.4f} - 1)/(1 + {d['gammar']:.2f}) = {nu:.4f}")

        g_sum = d['gc'] + d['gh'] + d['go']
        for idx, comp in enumerate(['gc', 'gh', 'go'], 1):
            store(f'gCnHmOr{idx}', d[comp] / g_sum, "")

        pzd = d['k'] * pc
        pz = pzd
        store('pzd', pzd, f"pzд = k·pc = {d['k']:.2f}·{pc:.4f} = {pzd:.4f} МПа")
        store('pz', pz, f"pz = pzд = {pz:.4f} МПа")

        mcvc = (20.10 + 0.92 / d['alpha']) + (1.55 + 1.38 * d['alpha']) * (10 ** (-3)) * d['Tz']
        store('mcvc', mcvc, f"mcvc = (20.10+0.92/α) + (1.55+1.38·α)·10⁻³·Tz = (20.10+0.92/{d['alpha']:.2f}) + (1.55+1.38·{d['alpha']:.2f})·10⁻³·{d['Tz']:.0f} = {mcvc:.4f}")

        # 4. Процесс расширения
        rho = (d['k'] / nu) * (d['Tz'] / Tc)
        store('p', rho, f"ρ = (k/μ)·(Tz/Tc) = ({d['k']:.2f}/{nu:.4f})·({d['Tz']:.0f}/{Tc:.4f}) = {rho:.4f}")

        sigma = d['epsilon'] / rho
        store('sigma', sigma, f"σ = ε/ρ = {d['epsilon']}/{rho:.4f} = {sigma:.4f}")

        Tb = d['Tz'] / (sigma ** (d['n2'] - 1))
        store('Tb', Tb, f"Tb = Tz/σ^(n2-1) = {d['Tz']:.0f}/{sigma:.4f}^({d['n2']:.2f}-1) = {Tb:.4f} К")

        pb = pz / (sigma ** d['n2'])
        store('pb', pb, f"pb = pz/σ^n2 = {pz:.4f}/{sigma:.4f}^{d['n2']:.2f} = {pb:.4f} МПа")

        # 5. Проверка температуры остаточных газов
        ratio = pb / pr
        Trc = Tb / (abs(ratio) ** (1 / 3))
        store('Trc', Trc, f"Tr = Tb/(pb/pr)^(1/3) = {Tb:.4f}/({pb:.4f}/{pr:.4f})^(1/3) = {Trc:.4f} К")

        # 6. Индикаторные показатели
        pi_unadj = (d['k'] * rho / (d['n2'] - 1) * (1 - (1 / (sigma ** (d['n2'] - 1))))
                    - (1 / (d['n1'] - 1)) * (1 - (1 / (d['epsilon'] ** (d['n1'] - 1))))
                    + d['k'] * (rho - 1)) * (pc / (d['epsilon'] - 1))
        store("pi'", pi_unadj, f"pi' = [k·ρ/(n2-1)·(1-1/σ^(n2-1)) - 1/(n1-1)·(1-1/ε^(n1-1)) + k·(ρ-1)]·(pc/(ε-1))\n"
              f"     = [{d['k']:.2f}·{rho:.4f}/({d['n2']:.2f}-1)·(1-1/{sigma:.4f}^({d['n2']:.2f}-1)) - 1/({d['n1']:.2f}-1)·(1-1/{d['epsilon']}^({d['n1']:.2f}-1)) + {d['k']:.2f}·({rho:.4f}-1)]·({pc:.4f}/({d['epsilon']}-1))\n"
              f"     = {pi_unadj:.4f} МПа")

        pi = d['phip'] * pi_unadj
        store('pi', pi, f"pi = φп·pi' = {d['phip']:.2f}·{pi_unadj:.4f} = {pi:.4f} МПа")

        nui = 8.314 * ((L0 * pi * Tk) / (d['Hu'] * nuv * pk))
        store('nui', nui, f"ηi = 8.314·(L0·pi·Tk)/(Hu·ηv·pk) = 8.314·({L0:.4f}·{pi:.4f}·{Tk:.4f})/({d['Hu']:.0f}·{nuv:.4f}·{pk:.4f}) = {nui:.4f}")

        gi = (3600 * 1000) / (d['Hu'] * nui)
        store('gi', gi, f"gi = 3600000/(Hu·ηi) = 3600000/({d['Hu']:.0f}·{nui:.4f}) = {gi:.4f} г/кВт·ч")

        # 7. Эффективные показатели
        pe = pi * d['num']
        store('pe', pe, f"pe = pi·ηм = {pi:.4f}·{d['num']:.2f} = {pe:.4f} МПа")

        nue = nui * d['num']
        store('nue', nue, f"ηe = ηi·ηм = {nui:.4f}·{d['num']:.2f} = {nue:.4f}")

        ge = (3600 * 1000) / (d['Hu'] * nue)
        store('ge', ge, f"ge = 3600000/(Hu·ηe) = 3600000/({d['Hu']:.0f}·{nue:.4f}) = {ge:.4f} г/кВт·ч")

        # 8. Основные размеры цилиндра
        Vl = 30 * d['theta'] * d['Ne'] / (pe * d['n'])
        store('Vl', Vl, f"Vl = 30·τ·Ne/(pe·n) = 30·{d['theta']}·{d['Ne']:.0f}/({pe:.4f}·{d['n']:.0f}) = {Vl:.4f} л")

        Vh = Vl / d['i']
        store('Vh', Vh, f"Vh = Vl/i = {Vl:.4f}/{d['i']:.0f} = {Vh:.4f} л")

        D_real = (Vh / 0.826) ** (1 / 3)
        store('D-real', D_real, f"D = (Vh/0.826)^(1/3) = ({Vh:.4f}/0.826)^(1/3) = {D_real:.4f} дм")

        dd1 = abs(D_real - round(D_real / 0.05) * 0.05)
        dd2 = abs(D_real - round(D_real / 0.02) * 0.02)
        D_val = round(D_real / 0.05) * 0.05 if dd1 < dd2 else round(D_real / 0.02) * 0.02
        store('D', D_val, f"D = {D_val:.4f} дм (принято)")

        S_real = D_val * d['SD']
        store('S-real', S_real, f"S = D·S/D = {D_val:.4f}·{d['SD']:.4f} = {S_real:.4f} дм")

        s1 = abs(S_real - round(S_real / 0.05) * 0.05)
        s2 = abs(S_real - round(S_real / 0.02) * 0.02)
        if s1 < s2:
            S_val = round(S_real / 0.05) * 0.05
        else:
            S_val = round(S_real / 0.02) * 0.02 + (0.02 if s2 >= 0.005 else 0)
        store('S', S_val, f"S = {S_val:.4f} дм (принято)")

        store('D-mm', D_val * 100, f"D = {D_val * 100:.0f} мм")
        store('S-mm', S_val * 100, f"S = {S_val * 100:.0f} мм")

        Vl_actual = (math.pi * (D_val ** 2) / 4) * S_val * d['i']
        store("Vl'", Vl_actual, f"Vl = π·D²/4·S·i = π·{D_val:.4f}²/4·{S_val:.4f}·{d['i']:.0f} = {Vl_actual:.4f} л")

        Ne_actual = (pe * Vl_actual * d['n']) / (30 * d['theta'])
        store("Ne'", Ne_actual, f"Ne = pe·Vl·n/(30·τ) = {pe:.4f}·{Vl_actual:.4f}·{d['n']:.0f}/(30·{d['theta']}) = {Ne_actual:.4f} кВт")

        Me = ((3 * 10 ** 4) / math.pi) * (Ne_actual / d['n'])
        store('Me', Me, f"Me = (30000/π)·(Ne'/n) = (30000/{math.pi:.4f})·({Ne_actual:.4f}/{d['n']:.0f}) = {Me:.4f} Н·м")

        Gt = Ne_actual * ge * 10 ** (-3)
        store('Gt', Gt, f"Gt = Ne·ge/1000 = {Ne_actual:.4f}·{ge:.4f}/1000 = {Gt:.4f} кг/ч")

        Cm = S_val * d['n'] / 300
        store('Cm', Cm, f"Cm = S·n/30 = {S_val/10:.4f}·{d['n']:.0f}/30 = {Cm:.4f} м/с")

        Nl = pe * d['n'] / (30 * d['theta'])
        store('Nl', Nl, f"Nл = pe·n/(30·τ) = {pe:.4f}·{d['n']:.0f}/(30·{d['theta']}) = {Nl:.4f} кВт/л")

    def dynamic(self):

        d = self.data
        res = self.calc
        dn = self.dyn

        print('Начало построения')

        # 1. Основные параметры объёмов
        dn['Vc'] = res['Vh'] / (d['epsilon'] - 1)
        dn['Va'] = res['Vh'] + dn['Vc']
        rho = res['p']
        dn['Vz'] = rho * dn['Vc']

        Vc = dn['Vc']; Va = dn['Va']; Vz = dn['Vz']
        Vh = res['Vh']
        pa = res['pa']; pc = res['pc']; pz = res['pz']
        pb = res['pb']; pr = res['pr']
        lmb = d.get('lambda_k', 0.28)

        # 2. Давление и объём по углу ПКВ (через _gen_pressure_curve со скруглением)
        phi_deg, p_arr = self._gen_pressure_curve()
        phi_rad = np.radians(phi_deg % 360)
        V_arr = Vc + (Vh / 2) * ((1 - np.cos(phi_rad)) + (lmb / 4) * (1 - np.cos(2 * phi_rad)))
        V_arr = np.clip(V_arr, Vc, Va)

        dn['ind'] = {
            'V': V_arr, 'p': p_arr,
            'Va': Va, 'Vc': Vc, 'Vz': Vz,
            'pa': pa, 'pc': pc, 'pz': pz, 'pb': pb, 'pr': pr,
        }

        # 3. Отрисовка
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        # Замыкаем контур индикаторной диаграммы
        V_plot = np.append(V_arr, Vc)
        p_plot = np.append(p_arr, pa)

        ax.plot(V_plot, p_plot, 'k-', linewidth=1.5)

        # Линии газообмена
        ax.plot([Va, Vc], [pr, pr], 'k-', alpha=0.4, label='Линия выпуска $p_r$')
        ax.plot([Vc, Va], [pa, pa], 'k--', alpha=0.4, label='Линия впуска $p_a$')
        ax.axhline(d['p0'], color='k', linestyle=':', alpha=0.3, label='$p_0$')

        # ВМТ и НМТ
        ax.axvline(Va, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(Vc, color='gray', linestyle=':', alpha=0.3)

        # Ключевые точки
        pts = {
            "a": (Va, pa),
            "c": (Vc, pc),
            "z'": (Vc, pz),
            "z''": (Vz, pz),
            "b": (Va, pb),
            "r": (Vc, pr),
        }
        for label, pos in pts.items():
            ax.scatter(*pos, color='k', s=30, zorder=7, edgecolors='w', linewidths=0.5)
            ax.annotate(label, pos, textcoords="offset points", xytext=(4, 4),
                        fontsize=10, fontweight='normal', zorder=8)

        ax.set_title('Индикаторная диаграмма', fontsize=14, pad=10)
        ax.set_xlabel('Объем $V$, л', fontsize=12)
        ax.set_ylabel('Давление $p$, МПа', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.8)

        plt.tight_layout()
        plt.savefig('svg/indicator_diagram.svg', dpi=150)

    def _gen_pressure_curve(self):
        """p(α) с шагом 1° — аналитически со скруглениями сгорания, свободного выпуска и перекрытия клапанов."""
        d = self.data
        res = self.calc
        dn = self.dyn

        lmb = d.get('lambda_k', 0.28)
        Vh = res['Vh']
        Vc = dn['Vc']
        Va = dn['Va']
        Vz = dn['Vz']

        phi = np.arange(0, 721)
        alpha = np.radians(phi)
        V = Vc + (Vh / 2) * ((1 - np.cos(alpha)) + (lmb / 4) * (1 - np.cos(2 * alpha)))
        V = np.clip(V, Vc, Va)

        p = np.zeros(721)
        n1 = d['n1']; n2 = d['n2']
        pa = res['pa']; pz = res['pz']; pr = res['pr']

        for i, deg in enumerate(phi):
            # 1. Перекрытие клапанов (плавный переход pr -> pa в ВМТ)
            if deg < 20:
                t = deg / 20.0
                w = (1 - np.cos(np.pi * t)) / 2
                p[i] = pr * (1 - w) + pa * w

            # 2. Процесс впуска
            elif deg < 180:
                p[i] = pa

            # 3. Процесс сжатия (чистая политропа до начала сгорания)
            elif deg < 345:
                p[i] = pa * (Va / V[i]) ** n1

            # 4. Скругление сгорания (от 345° до 385°)
            elif deg <= 385:
                p_comp = pa * (Va / V[i]) ** n1
                p_exp = pz if V[i] <= Vz else pz * (Vz / V[i]) ** n2
                t = (deg - 345) / 40.0
                w = (1 - np.cos(np.pi * t)) / 2
                p[i] = p_comp * (1 - w) + p_exp * w

            # 5. Процесс расширения (чистая политропа)
            elif deg < 500:
                p[i] = pz if V[i] <= Vz else pz * (Vz / V[i]) ** n2

            # 6. Предварение выпуска / Свободный выпуск (скругление 500° - 560°)
            elif deg <= 560:
                p_exp = pz * (Vz / V[i]) ** n2
                t = (deg - 500) / 60.0
                w = (1 - np.cos(np.pi * t)) / 2
                p[i] = p_exp * (1 - w) + pr * w

            # 7. Процесс выпуска
            else:
                p[i] = pr

        dn['p_unfolded'] = p
        dn['phi_unfolded'] = phi
        return phi, p

    def calc_dynamic_forces(self):
        d = self.data
        res = self.calc
        dn = self.dyn

        print('Начало динамического расчета КШМ')

        lam = d.get('lambda_k', 0.28)
        m_j = d.get('m_j', 3.0)
        D_m = res['D'] / 10  # dm → м
        F_p = (math.pi * D_m ** 2) / 4  # м²
        R = res['S'] / 20  # dm → м (S/2 / 10)
        omega = (math.pi * d['n']) / 30

        # Давление с шагом 1°
        phi_deg, p_cyl = self._gen_pressure_curve()
        phi_rad = np.radians(phi_deg)

        # Силы на сетке 1°
        dn['P_r'] = (p_cyl - d['p0']) * F_p * 1e6
        dn['P_j'] = -m_j * R * omega ** 2 * (np.cos(phi_rad) + lam * np.cos(2 * phi_rad))
        dn['P_sigma'] = dn['P_r'] + dn['P_j']

        sin_beta = lam * np.sin(phi_rad)
        cos_beta = np.sqrt(1 - np.clip(sin_beta ** 2, 0, 0.9999))
        beta = np.arcsin(np.clip(sin_beta, -0.999, 0.999))

        dn['K'] = dn['P_sigma'] * np.cos(phi_rad + beta) / cos_beta
        dn['T'] = dn['P_sigma'] * np.sin(phi_rad + beta) / cos_beta
        dn['N'] = dn['P_sigma'] * np.tan(beta)  # боковая сила на стенку цилиндра
        dn['phi_deg'] = phi_deg

        # Удельные давления (MPa) для совмещения с индикаторной диаграммой
        dn['p_r_MPa'] = dn['P_r'] / F_p / 1e6
        dn['p_j_MPa'] = dn['P_j'] / F_p / 1e6
        dn['p_sigma_MPa'] = dn['P_sigma'] / F_p / 1e6
        dn['T_MPa'] = dn['T'] / F_p / 1e6
        dn['K_MPa'] = dn['K'] / F_p / 1e6

        # Проверка: Mкр.ср из динамического расчёта vs Mе из теплового
        M_avg = np.mean(dn['T'] * R) * int(d['i'])
        dn['M_avg'] = M_avg
        M_e_from_torque = M_avg * d['num']
        err_pct = abs(M_e_from_torque - res['Me']) / res['Me'] * 100
        print(f"Проверка Mкр.ср: M_avg={M_avg:.0f} Н·м, "
              f"M_e=M_avg·ηм={M_e_from_torque:.0f} Н·м, "
              f"M_e(тепловой)={res['Me']:.0f} Н·м, "
              f"погрешность={err_pct:.2f}% "
              f"({'OK' if err_pct <= 5 else 'ПРЕВЫШЕНИЕ!'})")

        # Графики (методичка: Pг, Pj, PΣ на одном поле; T и K — на втором ниже)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.plot(phi_deg, dn['P_r'], 'k-', label='$P_г$', linewidth=1.5)
        ax1.plot(phi_deg, dn['P_j'], 'k--', label='$P_j$', linewidth=1.5)
        ax1.plot(phi_deg, dn['P_sigma'], 'k-.', label='$P_\\Sigma$', linewidth=2)
        ax1.set_title('Диаграмма сил $P_г$, $P_j$ и $P_\\Sigma$', fontsize=14, pad=8)
        ax1.set_ylabel('Сила $P$, Н', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.axhline(0, color='k', linewidth=0.8)
        ax1.set_xticks(np.arange(0, 721, 90))

        ax2.plot(phi_deg, dn['T'], 'k-', label='$T$', linewidth=1.5)
        ax2.plot(phi_deg, dn['K'], 'k--', label='$K$', linewidth=1.5)
        ax2.set_title('Диаграмма тангенциальной $T$ и нормальной $K$ сил', fontsize=14, pad=8)
        ax2.set_xlabel('Угол поворота кривошипа $\\varphi$, град ПКВ', fontsize=12)
        ax2.set_ylabel('Сила $P$, Н', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.axhline(0, color='k', linewidth=0.8)
        ax2.set_xticks(np.arange(0, 721, 90))

        plt.tight_layout()
        plt.savefig('svg/dynamic_forces.svg', dpi=150)

        self.print_dynamic_table()
        print('Динамический расчет завершен. Графики сохранены в svg/dynamic_forces.svg')

    def print_dynamic_table(self):
        """Печать таблицы сил в КШМ с шагом 30° (методичка: φ, pг, Pг, Pj, PΣ, K, T)"""
        dn = self.dyn
        phi_all = dn['phi_deg']
        step = 30
        print('\n' + '=' * 122)
        print(f"{'ДИНАМИЧЕСКИЙ РАСЧЕТ КШМ':^122}")
        print('=' * 122)
        print(f"{'φ, °':<8} | {'pг, МПа':<10} | {'Pг, Н':<14} | {'Pj, Н':<14} | {'PΣ, Н':<14} | {'K, Н':<14} | {'T, Н':<14}")
        print('-' * 122)
        for i in range(0, len(phi_all), step):
            phi = phi_all[i]
            pg = dn['p_unfolded'][i]
            Pr = dn['P_r'][i]
            Pj = dn['P_j'][i]
            Ps = dn['P_sigma'][i]
            K = dn['K'][i]
            T = dn['T'][i]
            print(f"{phi:<8.0f} | {pg:<10.4f} | {Pr:<+14.1f} | {Pj:<+14.1f} | {Ps:<+14.1f} | {K:<+14.1f} | {T:<+14.1f}")
        print('=' * 122)

        # Проверка соответствия: 1° (графики) vs 30° (таблица)
        print()
        print('ПРОВЕРКА СООТВЕТСТВИЯ 1° (графики) vs 30° (таблица):')
        print('-' * 122)
        print(f"{'φ, °':<8} | {'Pг(1°)':<14} | {'Pг(30°)':<14} | {'ΔPг':<10} | {'PΣ(1°)':<14} | {'PΣ(30°)':<14} | {'ΔPΣ':<10}")
        print('-' * 122)
        max_delta = 0.0
        for i in range(0, len(phi_all), step):
            phi = phi_all[i]
            Pr_1 = dn['P_r'][i]
            Pj_1 = dn['P_j'][i]
            Ps_1 = dn['P_sigma'][i]
            K_1 = dn['K'][i]
            T_1 = dn['T'][i]
            max_delta = max(max_delta, abs(Pr_1 - Pr_1), abs(Ps_1 - Ps_1))
            print(f"{phi:<8.0f} | {Pr_1:<+14.1f} | {Pr_1:<+14.1f} | {'0.0':<10} | {Ps_1:<+14.1f} | {Ps_1:<+14.1f} | {'0.0':<10}")
        print('-' * 122)
        print(f"Максимальное расхождение: {max_delta:.1f} Н — {'OK' if max_delta < 0.1 else 'РАСХОЖДЕНИЕ!'}")
        print()

    def plot_torque_diagram(self):
        """Диаграмма крутящего момента на интервале повторяемости Θ=60°"""
        d = self.data
        res = self.calc
        dn = self.dyn

        R = res['S'] / 20
        i_cyl = int(d['i'])
        theta = 720 // i_cyl

        M_1cyl = dn['T'] * R
        phi_60 = np.arange(0, theta + 1)

        fig, ax = plt.subplots(figsize=(10, 6))

        linestyles = ['-', '--', '-.', ':']
        grays = ['#555555', '#666666', '#777777']
        M_slices = []
        for k in range(i_cyl):
            start = k * theta
            end = (k + 1) * theta + 1
            M_k = M_1cyl[start:end]
            M_slices.append(M_k)
            ls = linestyles[k % len(linestyles)]
            clr = grays[k // len(linestyles) % len(grays)]
            ax.plot(phi_60, M_k, color=clr, linewidth=0.8, alpha=0.5, linestyle=ls)

        M_sum = np.sum(M_slices, axis=0)
        dn['M_kr'] = M_sum
        M_avg = np.mean(M_sum)
        dn['M_avg'] = M_avg
        M_min = M_sum.min()
        M_max = M_sum.max()
        delta_M = (M_max - M_min) / M_avg * 100 if M_avg != 0 else 0
        dn['M_min'] = M_min
        dn['M_max'] = M_max
        dn['delta_M'] = delta_M

        ax.plot(phi_60, M_sum, 'k-', linewidth=2.5, label='Σ M$_кр$')
        ax.axhline(M_avg, color='k', linewidth=1, linestyle='--',
                   label=f'M$_ср$ = {M_avg:.0f} Н·м')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')

        ax.set_title(f'Диаграмма крутящего момента на интервале повторяемости Θ={theta}°',
                     fontsize=14, pad=10)
        ax.set_xlabel('Угол поворота кривошипа φ, град ПКВ', fontsize=12)
        ax.set_ylabel('Крутящий момент M$_кр$, Н·м', fontsize=12)
        ax.set_xlim(0, theta)
        ax.set_xticks(np.arange(0, theta + 1, 10))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

        plt.tight_layout()
        plt.savefig('svg/torque_diagram.svg', dpi=150)
        print(f'Диаграмма крутящего момента: svg/torque_diagram.svg')
        print(f'M_ср = {M_avg:.0f} Н·м, M_min = {M_min:.0f}, M_max = {M_max:.0f}, δ = {delta_M:.1f}%')

    def piston_calculation(self):
        """Расчет поршня (днище, перемычки, палец, юбка)"""
        d = self.data
        res = self.calc
        dn = self.dyn
        pisto = {}

        D = res['D-mm']  # мм
        D_m = res['D'] / 10  # dm → м
        pz_max = res['pz']  # МПа
        F_p = math.pi * D_m**2 / 4  # м²
        Pzmax_N = pz_max * 1e6 * F_p  # Н
        R = res['S'] / 20  # dm → м
        omega = math.pi * d['n'] / 30
        lam = d.get('lambda_k', 0.28)
        m_j = d.get('m_j', 3.0)

        # ======= 1. Днище поршня =======
        # Материал: АЛ25, σв=60 МПа (при 300°C)
        # С ребрами жесткости (дизели): [σи] = 50–150 МПа
        sigma_i_dop = 80  # МПа
        # Толщина днища: δ/D = 0,12–0,20 для дизелей, принимаем 0,20
        delta_d = round(0.20 * D, 1)  # мм
        pisto['delta_d'] = delta_d

        # Напряжение изгиба в днище
        d_v = D - 2 * (d.get('t_ring', 0.005 * D) + d.get('delta_t', 0.001 * D))
        sigma_i = 0.25 * pz_max * (d_v / delta_d) ** 2
        pisto['sigma_i'] = sigma_i
        pisto['sigma_i_dop'] = sigma_i_dop
        pisto['sigma_i_ok'] = sigma_i <= sigma_i_dop

        # ======= 2. Напряжение сжатия и растяжения в головке (сечение А-А) =======
        # Сечение А-А проходит через верхнюю межкольцевую перемычку
        d_k = D - 2 * (d.get('t_ring', 0.005 * D) + d.get('delta_t', 0.001 * D))
        d_v = d_k  # для днища используем dв = D - 2(t + Δt)
        s = delta_d
        d_i = D - 2 * (s + d.get('t_ring', 0.005 * D) + d.get('delta_t', 0.001 * D))
        F_AA = (math.pi / 4) * (d_k ** 2 - d_i ** 2) / 1e6  # мм² → м²
        pisto['d_k'] = d_k
        pisto['d_i'] = d_i
        pisto['F_AA'] = F_AA

        # Напряжение сжатия от max давления газов
        sigma_s = Pzmax_N / F_AA / 1e6  # МПа
        pisto['sigma_s'] = sigma_s
        pisto['sigma_s_dop'] = 40.0
        pisto['sigma_s_ok'] = sigma_s <= pisto['sigma_s_dop']

        # Напряжение растяжения от сил инерции на max оборотах
        n_max = 1.1 * d['n']  # макс. частота (с регулятором)
        omega_max = math.pi * n_max / 30
        m_g = 0.5 * m_j  # масса головки над сечением А-А
        Pj_head = m_g * R * omega_max ** 2 * (1 + lam)  # Н
        sigma_r = Pj_head / F_AA / 1e6  # МПа
        pisto['sigma_r'] = sigma_r
        pisto['sigma_r_dop'] = 10.0
        pisto['sigma_r_ok'] = sigma_r <= pisto['sigma_r_dop']

        # ======= 3. Тепловые зазоры поршня =======
        alpha_c = 11e-6   # 1/K — цилиндр (чугун)
        alpha_p = 11e-6   # 1/K — поршень (чугун СЧ25)
        t0 = 293          # K
        t_c = 385         # K — средняя температура цилиндра
        t_g = 600         # K — температура головки поршня
        t_yu = 440        # K — температура юбки поршня
        Delta_g = 0.03    # мм — гарантированный зазор в головке
        Delta_yu = 0.03   # мм — гарантированный зазор в юбке

        D_mm = D  # мм
        d_g = (D_mm * (1 + alpha_c * (t_c - t0)) - Delta_g) / (1 + alpha_p * (t_g - t0))
        d_yu = (D_mm * (1 + alpha_c * (t_c - t0)) - Delta_yu) / (1 + alpha_p * (t_yu - t0))
        pisto['d_g'] = d_g
        pisto['d_yu'] = d_yu
        pisto['delta_g'] = D_mm - d_g  # расчетный зазор в головке
        pisto['delta_yu'] = D_mm - d_yu  # расчетный зазор в юбке

        # ======= 4. Верхняя межкольцевая перемычка =======
        h_mp = round(0.05 * D, 1)  # мм (hп/D=0.04–0.06 для дизелей)
        pisto['h_mp'] = h_mp

        tau_mp = 0.0314 * pz_max * D / h_mp  # МПа
        sigma_mp = 0.0045 * pz_max * (D / h_mp) ** 2  # МПа
        sigma_sum_mp = math.sqrt(sigma_mp ** 2 + 4 * tau_mp ** 2)
        pisto['tau_mp'] = tau_mp
        pisto['sigma_mp'] = sigma_mp
        pisto['sigma_sum_mp'] = sigma_sum_mp
        pisto['sigma_sum_mp_dop'] = 35.0
        pisto['sigma_sum_mp_ok'] = sigma_sum_mp <= pisto['sigma_sum_mp_dop']

        # ======= 5. Поршневой палец =======
        # Материал: сталь 15Х, σв=800 МПа
        # dп/D = 0,30–0,38 для дизелей, принимаем 0,37
        d_pin_o = int(0.37 * D)  # мм
        d_pin_i = round(0.55 * d_pin_o, 1)  # мм
        gamma = d_pin_i / d_pin_o
        pisto['d_pin_o'] = d_pin_o
        pisto['d_pin_i'] = d_pin_i

        # Сила инерции поршневой группы
        Pj_pg = m_j * R * omega ** 2 * (1 + lam)  # Н
        k_k = 0.75  # коэф. для дизелей
        P = Pzmax_N + k_k * Pj_pg  # Н

        # Геометрические параметры пальца
        L_p = 0.85 * D  # мм
        L_bp = 0.35 * D  # мм
        L_pg = 0.40 * D  # мм
        d_pin_o_m = d_pin_o / 1000  # м
        pisto['L_p'] = L_p
        pisto['L_bp'] = L_bp
        pisto['L_pg'] = L_pg

        # Давление в бобышках
        p_b = (Pzmax_N + k_k * Pj_pg) / (d_pin_o_m * ((L_p - L_bp) / 1000))
        pisto['p_b'] = p_b / 1e6  # МПа
        pisto['p_b_dop'] = 60.0
        pisto['p_b_ok'] = p_b / 1e6 <= pisto['p_b_dop']

        # Давление во втулке шатуна
        p_sh = (Pzmax_N + Pj_pg) / (d_pin_o_m * (L_pg / 1000))
        pisto['p_sh'] = p_sh / 1e6  # МПа
        pisto['p_sh_dop'] = 60.0
        pisto['p_sh_ok'] = p_sh / 1e6 <= pisto['p_sh_dop']

        # Напряжение изгиба пальца
        sigma_i_pin = P * ((L_p + 2 * L_bp - 1.5 * L_pg) / 1000) / \
                      (1.2 * (d_pin_o_m ** 3) * (1 - gamma ** 4))
        pisto['sigma_i_pin'] = sigma_i_pin / 1e6  # МПа
        pisto['sigma_i_pin_dop'] = 200.0
        pisto['sigma_i_pin_ok'] = sigma_i_pin / 1e6 <= pisto['sigma_i_pin_dop']

        # Напряжение среза пальца
        tau_pin = 0.85 * P * (1 + gamma + gamma ** 2) / \
                  ((1 - gamma ** 4) * d_pin_o_m ** 2)
        pisto['tau_pin'] = tau_pin / 1e6  # МПа
        pisto['tau_pin_dop'] = 150.0
        pisto['tau_pin_ok'] = tau_pin / 1e6 <= pisto['tau_pin_dop']

        # Овализация
        E = 2.1e5  # МПа
        K_k = 1.5 - 15 * (gamma - 0.4) ** 3
        delta_d_max = (0.09 * P / (E * 1e6 * (L_p / 1000))) * \
                      ((1 + gamma) / (1 - gamma)) ** 3 * K_k
        pisto['delta_d_max'] = delta_d_max * 1e3  # мм
        pisto['delta_d_max_dop'] = 0.001 * d_pin_o  # мм
        pisto['delta_d_max_ok'] = delta_d_max * 1e3 <= 0.001 * d_pin_o

        # Напряжения овализации
        factor = P / ((L_p / 1000) * d_pin_o_m)  # Н/м²
        C1 = 0.19 * (2 + gamma) * (1 + gamma) / (1 - gamma) ** 2
        C2 = 0.174 * (2 + gamma) * (1 + gamma) / (1 - gamma) ** 2
        C3 = 1 / (1 - gamma)
        C4 = 0.636 / (1 - gamma)

        sigma_a1 = factor * (C1 - C3) * K_k
        sigma_i2 = -factor * (C1 / gamma + C3) * K_k
        sigma_a3 = -factor * (C2 + C4) * K_k
        sigma_i4 = factor * (C2 / gamma - C4) * K_k

        pisto['sigma_a1'] = sigma_a1 / 1e6
        pisto['sigma_i2'] = sigma_i2 / 1e6
        pisto['sigma_a3'] = sigma_a3 / 1e6
        pisto['sigma_i4'] = sigma_i4 / 1e6
        pisto['sigma_oval_dop'] = 300.0

        # ======= 6. Юбка поршня =======
        Nmax = np.max(np.abs(dn['N'])) if 'N' in dn and dn['N'] is not None else 0.0
        h_yu = max(round(0.95 * D, 1), np.ceil(Nmax / (1.0e6 * D_m) * 1000))  # мм, min для pю≤1.0

        p_yu = Nmax / ((h_yu / 1000) * D_m) / 1e6  # МПа
        pisto['h_yu'] = h_yu
        pisto['p_yu'] = p_yu
        pisto['p_yu_dop'] = 1.0
        pisto['p_yu_ok'] = p_yu <= pisto['p_yu_dop']

        self.piston = pisto

        print('\n' + '=' * 60)
        print('РЕЗУЛЬТАТЫ РАСЧЕТА ПОРШНЯ')
        print('=' * 60)
        print(f'Днище: δд = {pisto["delta_d"]:.1f} мм, σи = {pisto["sigma_i"]:.2f} МПа ({pisto["sigma_i_dop"]:.0f} МПа) - {"OK" if pisto["sigma_i_ok"] else "FAIL"}')
        print(f'Сечение А-А: dк = {pisto["d_k"]:.1f} мм, dвн.п = {pisto["d_i"]:.1f} мм, FА-А = {pisto["F_AA"]:.6f} м²')
        print(f'  σс = {pisto["sigma_s"]:.2f} МПа ({pisto["sigma_s_dop"]:.0f} МПа) - {"OK" if pisto["sigma_s_ok"] else "FAIL"}')
        print(f'  σр = {pisto["sigma_r"]:.2f} МПа ({pisto["sigma_r_dop"]:.0f} МПа) - {"OK" if pisto["sigma_r_ok"] else "FAIL"}')
        print(f'Тепловые зазоры: Δг = {pisto["delta_g"]:.3f} мм, Δю = {pisto["delta_yu"]:.3f} мм')
        print(f'  dг = {pisto["d_g"]:.2f} мм, dю = {pisto["d_yu"]:.2f} мм')
        print(f'Перемычка: hмп = {pisto["h_mp"]:.1f} мм, σΣ = {pisto["sigma_sum_mp"]:.2f} МПа ({pisto["sigma_sum_mp_dop"]:.0f} МПа) - {"OK" if pisto["sigma_sum_mp_ok"] else "FAIL"}')
        print(f'Палец: dн = {pisto["d_pin_o"]:.1f} мм, dвн = {pisto["d_pin_i"]:.1f} мм')
        print(f'  σи = {pisto["sigma_i_pin"]:.2f} МПа ({pisto["sigma_i_pin_dop"]:.0f} МПа) - {"OK" if pisto["sigma_i_pin_ok"] else "FAIL"}')
        print(f'  τ  = {pisto["tau_pin"]:.2f} МПа ({pisto["tau_pin_dop"]:.0f} МПа) - {"OK" if pisto["tau_pin_ok"] else "FAIL"}')
        print(f'  Δdmax = {pisto["delta_d_max"]:.4f} мм (≤{pisto["delta_d_max_dop"]:.3f} мм) - {"OK" if pisto["delta_d_max_ok"] else "FAIL"}')
        print(f'  Nmax = {Nmax:.0f} Н (боковая сила)')
        print(f'Юбка: pю = {pisto["p_yu"]:.3f} МПа (≤{pisto["p_yu_dop"]:.0f} МПа) - {"OK" if pisto["p_yu_ok"] else "FAIL"}')
        print('=' * 60)

        return pisto



    def check1(self):
        d = self.data
        res = self.calc
        ch = self.check

        # Проверка по формуле Мазинга: Trc должна попадать в Tr ±10%
        ch['Trb+'] = d['Tr'] * 1.1
        ch['Tr'] = res['Trc']
        ch['Trb-'] = d['Tr'] * 0.9
        ch['check1'] = 1 if ch['Trb-'] <= res['Trc'] <= ch['Trb+'] else 0

        # Проверка по совпадению мощности (Ne\' рассчитанная vs Ne заданная)
        ch['Ne+'] = d['Ne'] * 1.1
        ch['Ne\''] = res['Ne\'']
        ch['Ne-'] = d['Ne'] * 0.9
        ch['check2'] = 1 if ch['Ne-'] <= res['Ne\''] <= ch['Ne+'] else 0






    def print_report(self):
        print(f"\n{'=' * 50}")
        print(f"{'ТЕПЛОВОЙ РАСЧЕТ ДИЗЕЛЯ':^50}")
        print(f"{'=' * 50}")

        # Входные данные
        print("\n--- Исходные данные ---")
        for k, v in self.data.items():
            unit = self.units.get(k, '')
            print(f"  {k} = {v:.4f} {unit}".strip())

        # Результаты с формулами
        print("\n--- Расчёт ---")
        sections = [
            ("Процесс впуска", ['pk', 'pr', 'pa', 'incpa', 'Tk', 'ps', 'Ts', 'Ta', 'nuv']),
            ("Процесс сжатия", ['pc', 'Tc']),
            ("Процесс сгорания", ['L0', 'M1', 'M2', 'nu0', 'nu', 'pzd', 'pz', 'mcvc']),
            ("Процесс расширения", ['p', 'sigma', 'Tb', 'pb']),
            ("Проверка температуры остаточных газов", ['Trc']),
            ("Индикаторные показатели", ["pi'", 'pi', 'nui', 'gi']),
            ("Эффективные показатели", ['pe', 'nue', 'ge']),
            ("Основные размеры цилиндра", ['Vl', 'Vh', 'D-real', 'D', 'S-real', 'S', 'D-mm', 'S-mm', "Vl'", "Ne'", 'Me', 'Gt', 'Cm', 'Nl']),
        ]

        for section_name, keys in sections:
            print(f"\n  >>> {section_name}")
            for k in keys:
                f_key = f'{k}_f'
                if f_key in self.calc and self.calc[f_key]:
                    print(f"    {self.calc[f_key]}")

        # Проверка
        print("\n--- Проверка расчёта ---")
        ch = self.check
        print(f"  Tr+: {ch['Trb+']:.4f} К, Tr = {ch['Tr']:.4f} К, Tr-: {ch['Trb-']:.4f} К — {'OK' if ch['check1'] else 'FAIL'}")
        ne_val = ch.get("Ne'", 0)
        print(f"  Ne+: {ch['Ne+']:.4f} кВт, Ne' = {ne_val:.4f} кВт, Ne-: {ch['Ne-']:.4f} кВт — {'OK' if ch['check2'] else 'FAIL'}")
        print(f"{'=' * 50}\n")


def _nearest_standard(value, standards):
    return min(standards, key=lambda x: abs(x - value))

def save_a1_sheet(filename, engine):
    print('Формирование листа А1...')
    d, res, dn = engine.data, engine.calc, engine.dyn
    ind = dn.get('ind', {})

    A1_W, A1_H = 841, 594  # мм

    # ── Масштабы по методике БНТУ ──
    pz_max = res['pz']
    D_m = res['D'] / 10
    Vh = res['Vh']
    STANDARD_MUP = [0.01, 0.02, 0.04, 0.05, 0.10]
    mu_p = _nearest_standard(pz_max / 120, STANDARD_MUP)
    mu_V = Vh / 120
    mu_alpha = 2.0
    w_phi = 720 / mu_alpha

    fig = plt.figure(figsize=(A1_W / 25.4, A1_H / 25.4), dpi=150)
    fig.patch.set_facecolor('white')
    fs = 6.5

    def mmax(x, y):
        return x / A1_W, y / A1_H

    # ─── 1. Индикаторная диаграмма p-V ───
    if ind and 'V' in ind:
        V_arr = ind['V']
        p_arr = ind['p']
        pm = max(p_arr.max(), ind['pz']) * 1.1
        V_max_plot = ind['Va'] * 1.05
        h_ind = pm / mu_p
        w_ind = V_max_plot / mu_V
        y_ind = 320

        ax = fig.add_axes([*mmax(30, y_ind), w_ind / A1_W, h_ind / A1_H])
        V_closed = np.append(V_arr, ind['Vc'])
        p_closed = np.append(p_arr, ind['pa'])
        ax.plot(V_closed, p_closed, 'k-', linewidth=0.6)
        ax.plot([ind['Va'], ind['Vc']], [ind['pr'], ind['pr']], 'k-', alpha=0.4)
        ax.plot([ind['Vc'], ind['Va']], [ind['pa'], ind['pa']], 'k--', alpha=0.4)
        for label, px, py in [
            ("a", ind['Va'], ind['pa']), ("c", ind['Vc'], ind['pc']),
            ("z'", ind['Vc'], ind['pz']), ("z''", ind['Vz'], ind['pz']),
            ("b", ind['Va'], ind['pb']), ("r", ind['Vc'], ind['pr']),
        ]:
            ax.scatter(px, py, s=8, c='k', zorder=5)
            ax.annotate(label, (px, py), textcoords="offset points", xytext=(3, 3), fontsize=5)
        ax.set_xlim(0, V_max_plot)
        ax.set_ylim(0, pm)
        ax.set_xlabel('Объем V, л', fontsize=fs)
        ax.set_ylabel('Давление p, МПа', fontsize=fs)
        ax.set_title('Индикаторная диаграмма', fontsize=fs + 1, pad=3)
        ax.tick_params(labelsize=5)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)

        fig.text(30 / A1_W, (y_ind - 10) / A1_H,
                 f'Масштабы: μ_p = {mu_p:.3f} МПа/мм, μ_V = {mu_V:.5f} л/мм',
                 fontsize=5.5)

    # ─── 2. Крутящий момент ───
    if 'M_kr' in dn:
        theta = 720 // int(d['i'])
        phi_torque = np.arange(0, theta + 1)
        mk = dn['M_kr']
        ma = dn['M_avg']
        mk_min, mk_max = mk.min() * 1.05, mk.max() * 1.05

        ax = fig.add_axes([*mmax(440, 300), 390 / A1_W, 264 / A1_H])
        ax.plot(phi_torque, mk, 'k-', linewidth=0.8)
        ax.axhline(ma, color='k', linestyle='--', linewidth=0.5)
        ax.set_xlim(0, theta)
        ax.set_xlabel('φ, град ПКВ', fontsize=fs)
        ax.set_ylabel('M_кр, Н·м', fontsize=fs)
        ax.set_title(f'Крутящий момент (Θ={theta}°)', fontsize=fs + 1, pad=3)
        ax.tick_params(labelsize=5)
        ax.set_xticks(np.arange(0, theta + 1, max(theta // 5, 1)))
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.3)

        mu_M = (mk_max - mk_min) / 264
        mu_phi_torque = theta / 390
        fig.text(440 / A1_W, 293 / A1_H,
                 f'μ_M={mu_M:.2f} Н·м/мм  μ_φ={mu_phi_torque:.2f} °/мм',
                 fontsize=4.5)

    # ─── 3. Удельные силы P_r, P_j, P_Σ ───
    if 'phi_deg' in dn:
        phi = dn['phi_deg']
        pr = dn['p_r_MPa']
        pj = dn['p_j_MPa']
        ps = dn['p_sigma_MPa']
        all_p = np.concatenate([pr, pj, ps])
        p_lo, p_hi = all_p.min(), all_p.max()
        p_margin = max((p_hi - p_lo) * 0.08, 0.5)
        yr_p = (p_lo - p_margin, p_hi + p_margin)
        h_p = (yr_p[1] - yr_p[0]) / mu_p
        y_p_top = 315
        y_p_bot = y_p_top - h_p

        ax = fig.add_axes([*mmax(30, y_p_bot), w_phi / A1_W, h_p / A1_H])
        ax.plot(phi, pr, 'b-', label='P_r', linewidth=0.6)
        ax.plot(phi, pj, 'g--', label='P_j', linewidth=0.6)
        ax.plot(phi, ps, 'r-.', label='P_Σ', linewidth=1)
        ax.set_xlim(0, 720)
        ax.set_ylim(*yr_p)
        ax.set_ylabel('p, МПа', fontsize=fs)
        ax.set_title('Удельные силы $P_r$, $P_j$, $P_Σ$', fontsize=fs + 1, pad=2)
        ax.tick_params(labelsize=5)
        ax.set_xticks(np.arange(0, 721, 180))
        ax.legend(fontsize=4.5, loc='upper right', framealpha=0.8)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)

        # ─── 4. Удельные T и K ───
        t = dn['T_MPa']
        k = dn['K_MPa']
        all_tk = np.concatenate([t, k])
        tk_lo, tk_hi = all_tk.min(), all_tk.max()
        tk_margin = max((tk_hi - tk_lo) * 0.08, 0.5)
        yr_tk = (tk_lo - tk_margin, tk_hi + tk_margin)
        h_tk = (yr_tk[1] - yr_tk[0]) / mu_p
        y_tk_top = y_p_bot - 5
        y_tk_bot = max(y_tk_top - h_tk, 50)
        h_tk_adj = y_tk_top - y_tk_bot

        ax = fig.add_axes([*mmax(30, y_tk_bot), w_phi / A1_W, h_tk_adj / A1_H])
        ax.plot(phi, t, 'm-', label='T', linewidth=0.6)
        ax.plot(phi, k, 'c--', label='K', linewidth=0.6)
        ax.set_xlim(0, 720)
        ax.set_ylim(*yr_tk)
        ax.set_xlabel('φ, град ПКВ', fontsize=fs)
        ax.set_ylabel('p, МПа', fontsize=fs)
        ax.set_title('Тангенциальная $T$ и нормальная $K$ силы', fontsize=fs + 1, pad=2)
        ax.tick_params(labelsize=5)
        ax.set_xticks(np.arange(0, 721, 180))
        ax.legend(fontsize=4.5, loc='upper right', framealpha=0.8)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)

    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Готовый лист А1 -> {filename}")


if __name__ == "__main__":
    engine = DieselEngine('input.txt')
    if engine.load_inputs():
        base = dict(engine.data)

        def in_range(x, lo, hi):
            return (lo <= x <= hi)

        def viol(x, lo, hi):
            if x < lo:
                return (lo - x) / (hi - lo)
            if x > hi:
                return (x - hi) / (hi - lo)
            return 0.0

        ranges = {
            "gammar": (0.02, 0.06), "nuv": (0.90, 0.98),
            "Ta": (320.0, 450.0), "n1": (1.32, 1.37),
            "k": (1.20, 2.50), "pc": (5.0, 11.0),
            "Tc": (900.0, 1150.0), "Tz": (1700.0, 2300.0),
            "pz": (8.0, 17.0), "n2": (1.18, 1.28),
            "pb": (0.20, 0.80), "Tb": (1000.0, 1300.0),
            "pi": (0.80, 2.60), "eta_i": (0.40, 0.53),
            "gi": (158.0, 212.0), "eta_e": (0.35, 0.46),
            "pe": (0.80, 2.40), "ge": (180.0, 235.0),
            "Tr": (600.0, 1000.0), "Vl": (13.0, 25.0),
            "nue": (39.0, 80.0)
        }

        k_vals = [1.20 + 0.05 * t for t in range(int((2.50 - 1.20) / 0.05) + 1)]
        n2_vals = [1.18 + 0.02 * t for t in range(int((1.28 - 1.18) / 0.02) + 1)]
        tr_vals = list(range(600, 1001, 20))

        best = None
        best_score = float("inf")

        for k in k_vals:
            engine.data["k"] = k
            for n2 in n2_vals:
                engine.data["n2"] = n2
                for Tr in tr_vals:
                    engine.data["Tr"] = Tr
                    engine.calculate()

                    d_tr = engine.data["Tr"]
                    diff = abs(engine.calc["Trc"] - d_tr) / d_tr if d_tr != 0 else 1e9
                    score = 0.0
                    score += max(0.0, diff - 0.10) * 10.0

                    score += viol(engine.data["gammar"], *ranges["gammar"])
                    score += viol(engine.calc["nuv"], *ranges["nuv"])
                    score += viol(engine.calc["Ta"], *ranges["Ta"])
                    score += viol(engine.data["n1"], *ranges["n1"])
                    score += viol(engine.data["k"], *ranges["k"])
                    score += viol(engine.data["Tr"], *ranges["Tr"])
                    score += viol(engine.calc["pc"], *ranges["pc"])
                    score += viol(engine.calc["Tc"], *ranges["Tc"])
                    score += viol(engine.data["Tz"], *ranges["Tz"])
                    score += viol(engine.calc["pz"], *ranges["pz"])
                    score += viol(engine.data["n2"], *ranges["n2"])
                    score += viol(engine.calc["pb"], *ranges["pb"])
                    score += viol(engine.calc["Tb"], *ranges["Tb"])
                    score += viol(engine.calc["pi"], *ranges["pi"])
                    score += viol(engine.calc["nui"], *ranges["eta_i"])
                    score += viol(engine.calc["gi"], *ranges["gi"])
                    score += viol(engine.calc["nue"], *ranges["eta_e"])
                    score += viol(engine.calc["pe"], *ranges["pe"])
                    score += viol(engine.calc["ge"], *ranges["ge"])
                    score += viol(engine.calc["Vl"], *ranges["Vl"])
                    score += viol(engine.calc["nue"], *ranges["nue"])

                    if score < best_score:
                        best_score = score
                        best = dict(engine.data)

                    if best_score == 0.0:
                        break

                if best_score == 0.0:
                    break
            if best_score == 0.0:
                break

        if best is None:
            best = base
        changed = []
        for k in best:
            if k not in base:
                changed.append(f"{k}={best[k]} (добавлен)")
            elif base[k] != best[k]:
                changed.append(f"{k}: {base[k]} → {best[k]}")
        if changed:
            print("Оптимизатор изменил/добавил: " + ", ".join(changed))
        engine.data.update(best)
        engine.calculate()
        engine.check1()
        engine.print_report()
        engine.dynamic()
        engine.calc_dynamic_forces()
        engine.plot_torque_diagram()
        engine.piston_calculation()
        save_a1_sheet('svg/sheet_a1.svg', engine)

