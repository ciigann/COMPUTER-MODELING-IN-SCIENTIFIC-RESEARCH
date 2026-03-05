import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import ttkbootstrap as ttk
from typing import Optional, Tuple, List

class SignificantDigitExtractor:
    """
    Класс для извлечения значащих цифр из числа.
    """

    @staticmethod
    def get_i_significant_digit(x: float, i: int) -> str:
        """
        Возвращает i-ю значащую цифру числа x.
        Если значащих цифр меньше, чем i, возвращает '-'.

        Args:
            x: Число, из которого извлекается цифра.
            i: Порядковый номер значащей цифры.

        Returns:
            str: i-я значащая цифра или '-', если её нет.
        """
        x_str = np.format_float_positional(x)
        k = 0
        flag = False
        if i + 1 > len(x_str):
            return "-"
        for j, char in enumerate(x_str):
            if char != "0" and not flag:
                flag = True
                if i == k:
                    return char
                if char != ".":
                    k += 1
            elif flag and char != ".":
                if k == i:
                    return char
                k += 1
        return "-"

class PhysicsModel:
    """
    Класс для физического моделирования движения тела.
    """

    def __init__(self):
        self.U = np.array([])  # Массив состояний: координаты и скорости
        self.E = np.array([])  # Массив энергий
        self.tau_1 = 0.0  # Шаг по времени для первого расчёта
        self.tau_2 = 0.0  # Шаг по времени для второго расчёта
        self.arr_accuracy = []  # Массив для точности расчётов

    def calculate_right_side(
        self, u: np.ndarray, g: float, mass: float, m: int
    ) -> np.ndarray:
        """
        Рассчитывает правую часть дифференциального уравнения для метода Эйлера.

        Args:
            u: Текущее состояние системы (координаты и скорости).
            g: Ускорение свободного падения.
            mass: Масса тела.
            m: Текущий шаг.

        Returns:
            np.ndarray: Правая часть уравнения.
        """
        f = np.zeros(4)
        f[0] = u[2]  # dx/dt = Vx
        f[1] = u[3]  # dy/dt = Vy
        f[2] = 0  # dVx/dt = 0 (без сопротивления воздуха)
        f[3] = -g  # dVy/dt = -g (ускорение свободного падения)

        # Рассчёт кинетической и потенциальной энергии
        self.E[m + 1, 0] = (f[0] ** 2 + f[1] ** 2) ** 0.5  # Скорость
        self.E[m + 1, 1] = (f[2] ** 2 + f[3] ** 2) ** 0.5  # Ускорение
        self.E[m + 1, 2] = (mass * self.E[m + 1, 0] ** 2) / 2 + mass * u[3] * g  # Полная энергия
        return f

    def simulate_trajectory(
        self,
        y_0: float,
        v_0: float,
        alpha: float,
        mass: float,
        M: int,
        tau: float,
    ) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
        """
        Моделирует траекторию движения тела методом Эйлера.

        Args:
            y_0: Начальная высота.
            v_0: Начальная скорость.
            alpha: Угол броска (в радианах).
            mass: Масса тела.
            M: Количество шагов.
            tau: Шаг по времени.

        Returns:
            Tuple[np.ndarray, Optional[Tuple[float, float]]]:
                - Массив состояний (координаты и скорости).
                - Точка пересечения с осью X (если есть).
        """
        x_0 = 0.0  # Начальная координата X
        g = 9.81  # Ускорение свободного падения

        # Инициализация массивов
        self.U = np.zeros((M + 1, 4))  # [x, y, Vx, Vy]
        self.E = np.zeros((M + 1, 3))  # [скорость, ускорение, полная энергия]

        # Начальные условия
        self.U[0, 0] = x_0
        self.U[0, 1] = y_0
        self.U[0, 2] = v_0 * np.cos(alpha)
        self.U[0, 3] = v_0 * np.sin(alpha)

        # Начальная энергия
        self.E[0, 0] = (self.U[0, 2] ** 2 + self.U[0, 3] ** 2) ** 0.5
        self.E[0, 1] = 0  # Ускорение в начальный момент
        self.E[0, 2] = (mass * self.E[0, 0] ** 2) / 2 + mass * y_0 * g

        crossing_point = None
        for m in range(M):
            # Метод Эйлера
            self.U[m + 1] = self.U[m] + tau * self.calculate_right_side(
                self.U[m], g, mass, m
            )

            # Проверка пересечения с осью X
            if self.U[m, 1] * self.U[m + 1, 1] <= 0 and m != 0:
                t = (0 - self.U[m, 1]) / (self.U[m + 1, 1] - self.U[m, 1])
                x_cross = self.U[m, 0] + t * (self.U[m + 1, 0] - self.U[m, 0])
                crossing_point = (x_cross, 0)
                break

        # Удаление нулевых значений
        self.U = self.U[self.U[:, 0] != 0.0]
        self.E = self.E[: len(self.U)]

        return self.U, crossing_point

    def calculate_accuracy(
        self,
        y_0: float,
        v_0: float,
        alpha: float,
        mass: float,
        M: int,
        dt2: float,
        dt1: int,
        tk: int,
    ) -> List[float]:
        """
        Оценивает точность расчётов координат и скорости.

        Args:
            y_0: Начальная высота.
            v_0: Начальная скорость.
            alpha: Угол броска.
            mass: Масса тела.
            M: Количество шагов.
            dt2: Шаг по времени для второго расчёта.
            dt1: Период расчётов.
            tk: Время для расчётов.

        Returns:
            List[float]: Точность для X, Y, Vx, Vy.
        """
        self.simulate_trajectory(y_0, v_0, alpha, mass, M, dt2 / 10)
        U_2 = self.U[:, [0, 1, 2, 3]]
        U_1 = self.U_1[:dt1]

        np.set_printoptions(suppress=True, precision=tk * 3)
        U_1 = np.array(U_1, dtype=float)
        U_2 = np.array(U_2[:dt1], dtype=float)

        arr = [round(self.dt1 / dt2)]
        for i in range(4):
            j = 0
            difference = True
            while difference and j != tk + 1:
                for l in range(dt1):
                    num_U_1 = np.format_float_positional(U_1[l, i], precision=10)
                    num_U_2 = np.format_float_positional(U_2[l, i], precision=10)

                    digit_U1 = SignificantDigitExtractor.get_i_significant_digit(
                        float(num_U_1), j
                    )
                    digit_U2 = SignificantDigitExtractor.get_i_significant_digit(
                        float(num_U_2), j
                    )
                    if digit_U1 != digit_U2 and digit_U1 != "-":
                        difference = False
                        break
                j += 1
            arr.append(np.format_float_positional(0.1 ** round(j - 1), precision=10))
        self.arr_accuracy.append(arr)
        return arr[1:]

class Plotter:
    """
    Класс для построения графиков.
    """

    @staticmethod
    def plot_trajectory(
        U: np.ndarray,
        fig_frame: tk.Frame,
        crossing_point: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Строит график траектории и отображает его в tkinter.

        Args:
            U: Массив состояний (координаты и скорости).
            fig_frame: Фрейм для отображения графика.
            crossing_point: Точка пересечения с осью X (опционально).
        """
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 8)
        ax.set_aspect("equal", adjustable="box")

        ax.plot(U[:, 0], U[:, 1])
        ax.set_title("Траектория движения тела")
        ax.set_xlabel("X, м")
        ax.set_ylabel("Y, м")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # Очистка фрейма и отображение графика
        for widget in fig_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        if crossing_point:
            print(f"Точка пересечения с осью X: {crossing_point[0]:.2f} м")

class ReportGenerator:
    """
    Класс для генерации отчётов.
    """

    @staticmethod
    def generate_report(U: np.ndarray, arr_accuracy: List[List[float]]) -> None:
        """
        Генерирует отчёт в виде текстовых файлов.

        Args:
            U: Массив состояний.
            arr_accuracy: Массив точности расчётов.
        """
        # Сохранение графика
        fig, ax = plt.subplots()
        ax.plot(U[:, 0], U[:, 1])
        ax.set_title("Траектория движения тела")
        ax.set_xlabel("X, м")
        ax.set_ylabel("Y, м")
        fig.savefig("trajectory.png")

        # Сохранение данных точности
        with open("accuracy_report.txt", "w", encoding="utf-8") as f:
            f.write("dt1/dt2\tX\tY\tVx\tVy\n")
            for row in arr_accuracy:
                f.write("\t".join(map(str, row)) + "\n")

        # Сохранение основных данных
        with open("simulation_report.txt", "w", encoding="utf-8") as f:
            f.write("x\t\ty\t\tVx\t\tVy\n")
            for row in U:
                f.write("\t".join(map(lambda x: f"{x:.6f}", row)) + "\n")

        print("Отчёт сохранён в файлах: trajectory.png, accuracy_report.txt, simulation_report.txt")

class SimulationApp:
    """
    Основной класс приложения с графическим интерфейсом.
    """

    def __init__(self, root: ttk.Window):
        self.root = root
        self.root.geometry("1700x1050")
        self.root.resizable(False, False)
        self.root.title("Моделирование движения тела")

        # Модели
        self.physics = PhysicsModel()
        self.plotter = Plotter()
        self.report_generator = ReportGenerator()

        # Фреймы
        self.fig_frame = tk.Frame(root)
        self.fig_frame.pack(side=tk.LEFT)

        self.entry_frame = tk.Frame(root)
        self.entry_frame.pack(side=tk.RIGHT)

        # Переменные для ввода
        self.entries = {
            "height": ttk.Entry(self.entry_frame, font=("Calibri", 14), width=20),
            "velocity": ttk.Entry(self.entry_frame, font=("Calibri", 14), width=20),
            "angle": ttk.Entry(self.entry_frame, font=("Calibri", 14), width=20),
            "mass": ttk.Entry(self.entry_frame, font=("Calibri", 14), width=20),
            "radius": ttk.Entry(self.entry_frame, font=("Calibri", 14), width=20),
            "step": ttk.Entry(self.entry_frame, font=("Calibri", 14), width=20),
            "steps_count": ttk.Entry(self.entry_frame, font=("Calibri", 14), width=20),
        }

        # Метки
        self.labels = {
            "height": ttk.Label(
                self.entry_frame, text="Высота (м):", font=("Calibri", 14, "bold")
            ),
            "velocity": ttk.Label(
                self.entry_frame, text="Начальная скорость (м/с):", font=("Calibri", 14, "bold")
            ),
            "angle": ttk.Label(
                self.entry_frame, text="Угол (градусы):", font=("Calibri", 14, "bold")
            ),
            "mass": ttk.Label(
                self.entry_frame, text="Масса (г):", font=("Calibri", 14, "bold")
            ),
            "radius": ttk.Label(
                self.entry_frame, text="Радиус (мм):", font=("Calibri", 14, "bold")
            ),
            "step": ttk.Label(
                self.entry_frame, text="Шаг по времени (с):", font=("Calibri", 14, "bold")
            ),
            "steps_count": ttk.Label(
                self.entry_frame, text="Количество шагов:", font=("Calibri", 14, "bold")
            ),
            "result": ttk.Label(self.entry_frame, text="", font=("Calibri", 14, "bold")),
        }

        # Кнопки
        self.buttons = {
            "simulate": ttk.Button(
                self.entry_frame,
                text="Запустить моделирование",
                command=self.run_simulation,
                style="primary.Outline.TButton",
            ),
            "report": ttk.Button(
                self.entry_frame,
                text="Сгенерировать отчёт",
                command=self.generate_report,
                style="primary.Outline.TButton",
            ),
        }

        # Размещение элементов
        self._place_widgets()

    def _place_widgets(self) -> None:
        """Размещает виджеты на форме."""
        for i, (key, label) in enumerate(self.labels.items()):
            label.pack(pady=5)
            self.entries[key].pack(pady=5)

        self.buttons["simulate"].pack(pady=20)
        self.labels["result"].pack(pady=10)
        self.buttons["report"].pack(pady=10)

    def run_simulation(self) -> None:
        """Запускает моделирование по введённым данным."""
        try:
            # Чтение входных данных
            height = float(self.entries["height"].get())
            velocity = float(self.entries["velocity"].get())
            angle = float(self.entries["angle"].get()) * np.pi / 180  # Перевод в радианы
            mass = float(self.entries["mass"].get())
            radius = float(self.entries["radius"].get())
            step = float(self.entries["step"].get())
            steps_count = int(self.entries["steps_count"].get())

            # Моделирование
            U, crossing_point = self.physics.simulate_trajectory(
                height, velocity, angle, mass, steps_count, step
            )

            # Построение графика
            self.plotter.plot_trajectory(U, self.fig_frame, crossing_point)

            # Сохранение результатов для отчёта
            self.physics.U_1 = U
            self.physics.dt1 = steps_count

            self.labels["result"]["text"] = "Моделирование завершено!"
        except ValueError as e:
            self.labels["result"]["text"] = f"Ошибка: {str(e)}"
        except Exception as e:
            self.labels["result"]["text"] = f"Неизвестная ошибка: {str(e)}"

    def generate_report(self) -> None:
        """Генерирует отчёт по результатам моделирования."""
        if not hasattr(self.physics, "U_1"):
            self.labels["result"]["text"] = "Сначала запустите моделирование!"
            return

        try:
            self.report_generator.generate_report(
                self.physics.U_1, self.physics.arr_accuracy
            )
            self.labels["result"]["text"] = "Отчёт успешно сгенерирован!"
        except Exception as e:
            self.labels["result"]["text"] = f"Ошибка при генерации отчёта: {str(e)}"

if __name__ == "__main__":
    root = ttk.Window(themename="yeti")
    app = SimulationApp(root)
    root.mainloop()
