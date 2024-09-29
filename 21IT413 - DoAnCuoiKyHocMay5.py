import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sns
import pickle
import requests
from tkinter import filedialog, messagebox, ttk
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk, Toplevel, Label, Button
from io import BytesIO
from urllib.request import urlopen


class RegressionApp:
    # Xử lý giao diện
   
    
    def __init__(self, root):
        self.root = root
        self.root.title("Mô hình hồi quy File CSV")
        self.root.geometry("1000x800")

        
        self.main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(expand=True, fill=tk.BOTH)

        self.controls_frame = tk.Frame(self.main_pane)
        self.main_pane.add(self.controls_frame)
    

        self.result_frame = tk.Frame(self.main_pane)
        self.main_pane.add(self.result_frame)

        self.result_chart = None  
        button_width = 20
        # Giao diện Button
        self.load_button = tk.Button(self.controls_frame, text="Chọn File CSV", width=15, command=self.load_csv)
        self.load_button.grid(row=0, column=0, padx=(60, 60), pady=6, sticky = "n")


        self.execute_button = tk.Button(self.controls_frame, text="Thực hiện", width=15, command=self.execute_regression)
        self.execute_button.grid(row=3, column=0, pady=6)

        self.handle_missing_button = tk.Button(self.controls_frame, text="Xử lý Data Missing Value", width=20, command=self.handle_missing_values)
        self.handle_missing_button.grid(row=4, column=0, pady=6)
        
        self.convert_data_button = tk.Button(self.controls_frame, text="Xử lý Convert Data", width=20, command=self.open_convert_window)
        self.convert_data_button.grid(row=5, column=0, pady=6)

        self.visualize_button = tk.Button(self.controls_frame, text="Visualize", width=15, command=self.visualize_relationship)
        self.visualize_button.grid(row=6, column=0, pady=6)

        self.recommend_button = tk.Button(self.controls_frame, text="Recommendation", command=self.open_recommendation_window)
        self.recommend_button.grid(row=7, column=0, pady=6)

        self.vars_frame = tk.Frame(self.controls_frame)
        self.vars_frame.grid(row=1, column=0)

        self.regression_frame = tk.Frame(self.controls_frame)
        self.regression_frame.grid(row=2, column=0)
        
        self.refresh_button = tk.Button(self.result_frame, text="Refresh", command=self.refresh_data)
        self.refresh_button.pack(side=tk.BOTTOM, pady=10)

        # Giao diện hiển thị thông tin Data từ File CSV
        self.result_text = tk.Text(self.result_frame, height=10, width=80)
        self.result_text.pack(expand=False , side=tk.BOTTOM, fill=tk.BOTH, pady=10)
        
        # Giao diện hiển thị thông tin sau khi dùng mô hình hồi quy
        self.result_text2 = tk.Text(self.result_frame, height=5, width=80)
        self.result_text2.pack(expand=True, side=tk.TOP, fill=tk.BOTH, pady=5)
        
        
        
        self.data = None
        self.target_var = tk.StringVar()
        self.input_vars = []
        self.regression_types = []
        self.selected_target_vars = []
        self.selected_input_vars = [] 
        self.viz_selected_target_vars = []
        self.viz_selected_input_vars = []
        self.plot_type_comboboxes = {}

        self.model_system = pickle.load(open('C:\\Users\\ASUS\\Downloads\\artifacts\\model.pkl','rb'))
        self.book_names = pickle.load(open('C:\\Users\\ASUS\\Downloads\\artifacts\\book_names.pkl','rb'))
        self.final_rating = pickle.load(open('C:\\Users\\ASUS\\Downloads\\artifacts\\final_rating.pkl','rb'))
        self.book_pivot = pickle.load(open('C:\\Users\\ASUS\\Downloads\\artifacts\\book_pivot.pkl','rb'))

        self.execute_button.grid_remove()
        self.handle_missing_button.grid_remove()
        self.convert_data_button.grid_remove()
        self.visualize_button.grid_remove()


        self.prediction_entry = None
        self.prediction_button = None

     
    def fetch_poster(self, suggestion):
        poster_urls = []
        for book_id in suggestion:
            book_name = self.book_pivot.index[book_id]
            match = self.final_rating[self.final_rating['title'] == book_name]
            if not match.empty:
                url = match.iloc[0]['image_url']  # Lấy URL từ bản ghi đầu tiên phù hợp
                poster_urls.append(url)
            else:
                poster_urls.append("No URL found")  # Phòng trường hợp sách không có URL
        return poster_urls

   
    def recommend_book(self, book_name):
        books_list = []
        try:
            book_id = np.where(self.book_pivot.index == book_name)[0][0]
        except IndexError:
            print(f"Book '{book_name}' not found in book_pivot.")
            return [], []
        
        distance, suggestion = self.model_system.kneighbors(self.book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)

        poster_url = self.fetch_poster(suggestion[0])
        
        for i in range(len(suggestion[0])):
            books = self.book_pivot.index[suggestion[0][i]]
            books_list.append(books)
        return books_list, poster_url

    def open_recommendation_window(self):
        recommendation_window = Toplevel(self.root)
        recommendation_window.title('Recommendation Window')
        recommendation_window.geometry("1000x800")

        self.entry = tk.Entry(recommendation_window)
        self.entry.pack()

        self.recommend_button = tk.Button(recommendation_window, text='Recommend', command=self.show_recommendations)
        self.recommend_button.pack()

        self.result_frame_recommendation = tk.Frame(recommendation_window)
        self.result_frame_recommendation.pack(pady=10)
    
    # Xử lý dữ liệu khi import file CSV vào chương trình
    def load_csv(self):
        
        # Yêu cầu người dùng chọn file CSV và lưu đường dẫn vào biến file_path
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        # Kiểm tra user đã chọn file CSV
        if not file_path:
            return
        # Đọc dữ liệu từ file CSV và lưu vào biến self.data
        self.data = pd.read_csv(file_path)

        self.selected_target_vars.clear()
        self.selected_input_vars.clear()
            
        # Tạo thông tin về dữ liệu để hiển thị trên giao diện
        info_text = f"Số dòng: {len(self.data)}\nSố cột: {len(self.data.columns)}\n\nKiểu dữ liệu của mỗi cột:\n"
        for column, dtype in self.data.dtypes.items():
            info_text += f"{column}: {dtype}\n"
        
         # Tính số lượng giá trị thiếu trong mỗi cột và thêm vào giao diện
        missing_values = self.data.isnull().sum()
        info_text += "\nSố lượng giá trị thiếu trong mỗi cột:\n"
        for column, missing_count in missing_values.items():
            info_text += f"{column}: {missing_count}\n"
        
        data_preview = "Một số dữ liệu từ file CSV:\n\n"
        data_preview += str(self.data) 
        info_text += "\n\n" + data_preview
        
        # Xóa nội dung hiện tại của result_text2 ( giao diện hiển thị ) và thêm thông tin mới
        self.result_text2.delete('1.0', tk.END)
        self.result_text2.insert(tk.END, info_text)
        
        self.show_variables()
        self.execute_button.grid()
        self.handle_missing_button.grid()
        self.convert_data_button.grid()
        self.visualize_button.grid()

    def open_recommendation_window(self):
        recommendation_window = Toplevel(self.root)
        recommendation_window.title('Test chuc nang Recommendation System')
        recommendation_window.geometry("1000x800")

 
        main_frame = tk.Frame(recommendation_window)
        main_frame.pack(expand=True, fill='both')

       
        input_frame = tk.Frame(main_frame)
        input_frame.grid(row=0, column=0, sticky='nw', pady=10, padx=10)

        label_title = tk.Label(input_frame, text=f"Chọn sách đã xem")
        label_title.pack(side='top', pady=5)
       
        self.book_name_var = tk.StringVar()
        self.book_name_combobox = ttk.Combobox(input_frame, textvariable=self.book_name_var, values=list(self.book_pivot.index))
        self.book_name_combobox.pack(side='top', padx=5, pady=5)

        
        self.recommend_button = tk.Button(input_frame, text='Thực hiện', command=self.show_recommendations)
        self.recommend_button.pack(side='top', padx=5, pady=5)

        
        self.result_frame_recommendation = tk.Frame(main_frame)
        self.result_frame_recommendation.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        main_frame.grid_columnconfigure(1, weight=1)  # Give more weight to the result frame


    def show_recommendations(self):
        book_name = self.book_name_var.get()
        if not book_name:
            messagebox.showerror("Error", "Please enter a book name.")
            return

        recommended_books, poster_urls = self.recommend_book(book_name)
        if not recommended_books:
            messagebox.showerror("Error", f"Book '{book_name}' not found.")
            return

        if recommended_books[0] == book_name:
            recommended_books = recommended_books[1:]
            poster_urls = poster_urls[1:]
            
        for widget in self.result_frame_recommendation.winfo_children():
            widget.destroy()

       
        grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, (book, url) in enumerate(zip(recommended_books, poster_urls)):
            book_frame = tk.Frame(self.result_frame_recommendation)
            book_frame.grid(row=grid_positions[i][0], column=grid_positions[i][1], sticky='nsew', padx=5, pady=5)
            self.result_frame_recommendation.grid_rowconfigure(grid_positions[i][0], weight=1)
            self.result_frame_recommendation.grid_columnconfigure(grid_positions[i][1], weight=1)

            label_title = tk.Label(book_frame, text=f"{book}", font=('Helvetica', 12, 'bold'))
            label_title.pack(side='top', pady=5)

            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    image_bytes = BytesIO(response.content)
                    image = Image.open(image_bytes)
                    
                    base_height = 300
                    h_percent = (base_height / float(image.size[1]))
                    w_size = int((float(image.size[0]) * float(h_percent)))
                    image = image.resize((w_size, base_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    image_label = tk.Label(book_frame, image=photo)
                    image_label.image = photo
                    image_label.pack(side='top')
            except Exception as e:
                print(f"Failed to load image from {url}: {e}")
                label_error = tk.Label(book_frame, text=f"Failed to load image from URL: {url}", font=('Helvetica', 12))
                label_error.pack(side='top')


    # Hàm làm mới dữ liệu
    def refresh_data(self):
        if self.data is None:
            messagebox.showerror("Lỗi", "Không có dữ liệu để làm mới.")
            return

        # Hiển thị lại thông tin dữ liệu từ file CSV
        info_text = f"Số dòng: {len(self.data)}\nSố cột: {len(self.data.columns)}\n\nKiểu dữ liệu của mỗi cột:\n"
        for column, dtype in self.data.dtypes.items():
            info_text += f"{column}: {dtype}\n"
        
        missing_values = self.data.isnull().sum()
        info_text += "\nSố lượng giá trị thiếu trong mỗi cột:\n"
        for column, missing_count in missing_values.items():
            info_text += f"{column}: {missing_count}\n"
        
        data_preview = "Một số dữ liệu từ file CSV:\n\n"
        data_preview += str(self.data)
        info_text += "\n\n" + data_preview
        
        self.result_text2.delete('1.0', tk.END)
        self.result_text2.insert(tk.END, info_text)

    def show_variables(self):
        # Xóa tất cả các biến đã có trong chọn biến mục tiêu và độc lập
        for widget in self.vars_frame.winfo_children():
            widget.destroy()
        for widget in self.regression_frame.winfo_children():
            widget.destroy()
        
        # Kiểm tra xem dữ liệu đã được tải hay chưa
        if self.data is None:
            messagebox.showerror("Lỗi", "Không thể load dữ liệu từ file CSV.")
            return
        
        tk.Label(self.vars_frame, text="Chọn biến mục tiêu:").pack()
        self.target_var = tk.StringVar()
        target_var_combobox = ttk.Combobox(self.vars_frame, textvariable=self.target_var, values=list(self.data.columns), state='readonly')
        target_var_combobox.pack()

        # Nút thêm biến mục tiêu vào danh sách
        add_target_button = tk.Button(self.vars_frame, text="Thêm biến mục tiêu", command=lambda: self.add_target(self.target_var.get()))
        add_target_button.pack()

        # Listbox hiển thị các biến mục tiêu đã chọn
        self.selected_target_listbox = tk.Listbox(self.vars_frame, height = 6)
        self.selected_target_listbox.pack()

        # Nút để xóa biến mục tiêu đã chọn
        remove_target_button = tk.Button(self.vars_frame, text="Xóa", command=self.remove_selected_target)
        remove_target_button.pack()

         # Tạo label và Listbox để chọn biến độc lập
        tk.Label(self.vars_frame, text="Chọn biến độc lập:").pack()
        self.input_var = tk.StringVar()
        input_var_combobox = ttk.Combobox(self.vars_frame, textvariable=self.input_var, values=list(self.data.columns), state='readonly')
        input_var_combobox.pack()

        add_input_button = tk.Button(self.vars_frame, text="Thêm biến độc lập", command=lambda: self.add_input(self.input_var.get()))
        add_input_button.pack()

        self.selected_vars_listbox = tk.Listbox(self.vars_frame, height = 6)
        self.selected_vars_listbox.pack()

        remove_input_button = tk.Button(self.vars_frame, text="Xóa", command=self.remove_selected_variable)
        remove_input_button.pack()

        # Tạo Checkbutton để chọn mô hình hồi quy
        tk.Label(self.regression_frame, text="Chọn mô hình hồi quy:").pack()
        self.regression_type_var = tk.StringVar(value="Hồi quy tuyến tính")
        self.regression_combobox = ttk.Combobox(self.regression_frame, textvariable=self.regression_type_var, values=["Hồi quy tuyến tính", "Hồi quy Logistic", "Hồi quy KNN", "Decision Tree", "Random Forest"], state='readonly')
        self.regression_combobox.pack()
            
         # Thêm ô nhập giá trị dự đoán
        tk.Label(self.regression_frame, text="Nhập giá trị cần dự đoán:").pack()
        self.predict_entry = tk.Entry(self.regression_frame)
        self.predict_entry.pack(pady=5)
        
    # Hàm thêm biến độc lập vào danh sách
    def add_input(self, input_var):
        if input_var and input_var not in self.selected_vars_listbox.get(0, tk.END):
            self.selected_vars_listbox.insert(tk.END, input_var)
            self.selected_input_vars.append(input_var)
    
     # Thêm biến mục tiêu vào Listbox
    def add_target(self, target):
        if target and target not in self.selected_target_listbox.get(0, tk.END):
            self.selected_target_listbox.insert(tk.END, target)
            self.selected_target_vars.append(target)

            
    def remove_selected_target(self):
        selected_indices = self.selected_target_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một biến mục tiêu để xóa.")
            return

        # Lấy chỉ mục của mục tiêu đã chọn
        selected_index = selected_indices[0]

        # Lấy tên biến mục tiêu từ Listbox
        selected_var = self.selected_target_listbox.get(selected_index)

        # Xóa tên biến mục tiêu khỏi danh sách Python `self.selected_target_vars`
        if selected_var in self.selected_target_vars:
            self.selected_target_vars.remove(selected_var)

        # Xóa tên biến mục tiêu khỏi Listbox
        self.selected_target_listbox.delete(selected_index)


    def update_selected_target_listbox(self):
            # Xóa danh sách biến đã chọn trước đó
        self.selected_target_listbox.delete(0, tk.END)
            
            # Thêm lại tất cả các biến mục tiêu đã chọn
        for var in self.selected_target_vars:
            self.selected_target_listbox.insert(tk.END, var)

        # Hàm xử lý xóa biến độc lập đã chọn
    def remove_selected_variable(self):
        selected_indices = self.selected_vars_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một biến độc lập để xóa.")
            return

        # Lấy chỉ mục của biến đã chọn
        selected_index = selected_indices[0]

        # Lấy giá trị biến đã chọn
        selected_var = self.selected_vars_listbox.get(selected_index)

        # Xóa biến đã chọn khỏi danh sách Python `self.selected_input_vars`
        if selected_var in self.selected_input_vars:
            self.selected_input_vars.remove(selected_var)

        # Xóa biến đã chọn khỏi Listbox
        self.selected_vars_listbox.delete(selected_index)

       
    def update_selected_vars_listbox(self):
            # Xóa danh sách biến đã chọn trước đó
        self.selected_vars_listbox.delete(0, tk.END)
            
            # Thêm lại tất cả các biến độc lập đã chọn
        for var in self.selected_input_vars:
            self.selected_vars_listbox.insert(tk.END, var)



    def plot_knn_elbow(self, X, y, max_k=10):
        # Tạo một list để lưu các giá trị MSE
        errors = []
        for i in range(1, max_k):
            knn = KNeighborsRegressor(n_neighbors=i)
            knn.fit(X, y)
            # Dự đoán giá trị của y trên dữ liệu huấn luyện
            pred_i = knn.predict(X)
             # Tính MSE và lưu vào list errors
            errors.append(np.mean((y - pred_i) ** 2))

        # Tạo một biểu đồ để hiển thị Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k), errors, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
        plt.title('Elbow Method chọn giá trị "k" tối ưu.')
        plt.xlabel('Số K Neighbors')
        plt.ylabel('Tổng lỗi bình phương')
        plt.show()




    def plot_knn_classification(self, X, y, n_neighbors=5):
         # Kiểm tra xem có 2 biến độc lập được chọn không
        if X.shape[1] != 2:
            raise ValueError("Yêu cầu chọn 2 biến độc lập vẽ biểu đồ ranh giới phân loại")
        
        # Đặt bước nhảy cho các chấm biểu thị cho biểu đồ
        h = .02 
        # Tạo màu sắc cho vùng phân loại
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        #Training biểu đồ
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X, y)

         # Xác định giới hạn của biểu đồ
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

         # Dự đoán nhãn cho từng điểm trên mặt phẳng
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)

         # Vẽ các điểm dữ liệu
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"3-Class classification (k = {n_neighbors})")
        plt.show()



    def execute_regression(self):
        # Kiểm tra xem đã load dữ liệu từ file CSV chưa
        if self.data is None:
            messagebox.showerror("Lỗi", "Không thể load dữ liệu từ file CSV.")
            return

         # Lấy tên biến mục tiêu và các mô hình hồi quy được chọn
        target = self.target_var.get()
        model_name = self.regression_type_var.get()


         # Lấy danh sách các biến độc lập được chọn
        selected_inputs = self.selected_input_vars  
        if len(selected_inputs) == 0:
            messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một biến độc lập.")
            return

         # Tạo các mảng numpy cho biến đầu vào và biến mục tiêu
        self.X = self.data[selected_inputs]
        self.y = self.data[target]

        # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        results = []
 

        # Hàm xử lý Hồi quy tuyến tính
        def linear_model():
            model = LinearRegression().fit(X_train, y_train)
            predictions = model.predict(X_test) #ReadingScore
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = sqrt(mse)
            r2 = r2_score(y_test, predictions) 
            results.append(f"Linear Regression Coef: {model.coef_}, Intercept: {model.intercept_}")
            results.append(f"- MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}, R^2={r2:.2f}")
            self.plot_results(X_test, y_test, predictions, f"{model_name} - Linear Regression", model=model, target_name=target, input_names=selected_inputs)
        
        # Hàm xử lý hồi quy Logistic
        def logistic_model():
            model = LogisticRegression().fit(X_train, y_train)
            predictions = model.predict(X_test)
            # Vẽ ma trận nhầm lẫn (confusion matrix) và hiển thị báo cáo phân loại
            cm = confusion_matrix(y_test, predictions)
            cr = classification_report(y_test, predictions)
            results.append(f"Ma trận nhầm lẫn:\n{cm}\nBáo cáo phân loại:\n{cr}")
            self.plot_results(X_test, y_test, predictions, f"{model_name} - Logistic Regression", model=model, target_name=target, input_names=selected_inputs)
            
        # Hàm xử lý hồi quy KNN
        def knn_model():
            X_train_selected = X_train[selected_inputs]
            X_test_selected = X_test[selected_inputs]
            if len(selected_inputs) == 1:
                    # Nếu chỉ có 1 biến độc lập, chọn tự chọn giá trị tối ưu cho K
                mse_for_diff_k = []
                for k in range(1, 10):
                    model = KNeighborsRegressor(n_neighbors=k)
                    model.fit(X_train_selected, y_train)
                    predictions = model.predict(X_test_selected)
                    mse = mean_squared_error(y_test, predictions)
                    mse_for_diff_k.append(mse)
                self.plot_knn_elbow(X_train_selected, y_train) 
                best_k = mse_for_diff_k.index(min(mse_for_diff_k)) + 1
                model = KNeighborsRegressor(n_neighbors=best_k)
                model.fit(X_train_selected, y_train)
                predictions = model.predict(X_test_selected)
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                rmse = sqrt(mse)
                results.append(f"{model_name} - Best K: {best_k} - MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                self.plot_results(X_test_selected, y_test, predictions, f"{model_name} - KNN Regression", model=model, target_name=self.target_var.get(), input_names=selected_inputs)
            elif len(selected_inputs) == 2:
                self.plot_knn_classification(X_train_selected.to_numpy(), y_train.to_numpy(), n_neighbors=5)
                self.plot_knn_elbow(X_train_selected, y_train) 
                
        # Hàm xử lý Decision Tree
        def decision_tree_model():
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            cm = confusion_matrix(y_test, predictions)
            results.append(f"Decision Tree Confusion Matrix:\n{cm}")
            self.plot_results(X_test, y_test, predictions, f"{model_name}", model=model, target_name=target, input_names=selected_inputs)
            
        # Hàm xử lý Random Forest
        def random_forest_model():
            if self.data is None:
                messagebox.showerror("Lỗi", "Data chưa được tải")
                return

            target = self.target_var.get()
            selected_inputs = self.selected_input_vars
            if not selected_inputs:
                messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một biến độc lập")
                return

            X = self.data[selected_inputs]
            y = self.data[target]


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Khởi tạo train model, X_train, y_train này nó vẫn là dữ liệu gốc chưa random
            # sau khi train gọi RandomForestClassifier thì nó sẽ random theo bootstrap
            rf_model = RandomForestClassifier(n_estimators=5, random_state=42)
            rf_model.fit(X_train, y_train)

            
            predictions = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # Đếm Voting
            unique, counts = np.unique(predictions, return_counts=True)
            voting_results = dict(zip(unique, counts))

            # Dự đoán xác suất trung bình
            if hasattr(rf_model, "predict_proba"):
                probabilities = rf_model.predict_proba(X_test)
                average_probabilities = np.mean(probabilities, axis=0)

            # Thêm các chỉ số vào giao diện
            results.append(f"Độ chính xác khi dự đoán mô hình Random Forest: {accuracy:.2f}")
            results.append("Kết quả Voting: ")
            results.append(str(voting_results))
            results.append("Kết quả xác suất trung bình: ")
            results.append(str(average_probabilities))
            results.append("Dự đoán trên tập thử nghiệm: ")
            results.append(str(predictions))


            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, "\n".join(results))

            # Vẽ cây quyết định
            for i, tree in enumerate(rf_model.estimators_):
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(tree, ax=ax, filled=True, feature_names=selected_inputs, class_names=np.unique(y).astype(str))
                ax.set_title(f'Tree {i+1}')
                plt.show()
                    
        

        if model_name == "Hồi quy tuyến tính":
            self.model = LinearRegression().fit(X_train, y_train)
            linear_model()
        elif model_name == "Hồi quy Logistic":
            self.model = LogisticRegression().fit(X_train, y_train)
            logistic_model()
        elif model_name == "Hồi quy KNN":
            self.model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
            knn_model()
        elif model_name == "Decision Tree":
            self.model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
            decision_tree_model()
        elif model_name == "Random Forest":
            self.model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_train, y_train)
            random_forest_model()
 
        
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, "\n".join(results))
        
        if hasattr(self, 'model'):
            self.execute_prediction()
        else:
            messagebox.showerror("Lỗi", "Mô hình chưa được khởi tạo.")



    def execute_prediction(self):
            if self.data is None or not hasattr(self, 'model'):
                messagebox.showerror("Lỗi", "Chưa có mô hình hoặc dữ liệu.")
                return
            
            input_text = self.predict_entry.get()
            try:
                predict_values = [float(val.strip()) for val in input_text.split(',')]
                
                if len(predict_values) != len(self.selected_input_vars):
                    messagebox.showerror("Lỗi nhập liệu", f"Bạn cần nhập đúng {len(self.selected_input_vars)} giá trị phân cách bởi dấu phẩy.")
                    return
                
                predict_values_array = np.array(predict_values).reshape(1, -1)
                prediction = self.model.predict(predict_values_array)
                
                self.plot_results(self.X, self.y, self.model.predict(self.X), "Hiển thị giá trị dự đoán", self.model, self.target_var.get(), self.selected_input_vars, prediction=prediction, input_value=predict_values)
                messagebox.showinfo("Kết quả dự đoán", f"Dự đoán cho đầu vào {input_text}: {prediction[0]}")
            except ValueError:
                messagebox.showerror("Lỗi nhập liệu", "Vui lòng nhập các số hợp lệ, phân cách nhau bởi dấu phẩy.")

    def plot_results(self, X, y, y_pred, title, model=None, target_name=None, input_names=None, prediction=None, input_value=None):
        fig, ax = plt.subplots()
        xlabel = input_names[0] if input_names else 'Biến độc lập'
        ylabel = target_name if target_name else 'Biến mục tiêu'
        
        def plot_linear_model(ax):
         # Nếu chỉ có một biến độc lập, vẽ biểu đồ scatter plot và đường hồi quy
            if len(input_names) == 1:
                ax.scatter(X.iloc[:, 0], y, color='blue', label='Giá trị thực tế')
                ax.plot(X.iloc[:, 0], y_pred, color='red', label='Giá trị dự đoán')
                if prediction is not None and input_value is not None:
                    ax.scatter(input_value[0], prediction, color='green', s=100, label='Giá trị dự đoán cụ thể', zorder=5)
                ax.set_xlabel(xlabel) 
                ax.set_xlabel(ylabel) 
                ax.set_title(title)
                ax.legend()
                plt.show()

            elif len(input_names) == 2:
                # Nếu có hai biến độc lập, vẽ biểu đồ scatter plot 3D
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X[input_names[0]], X[input_names[1]], y, color='blue', label='Giá trị thực tế')
                ax.scatter(X[input_names[0]], X[input_names[1]], y_pred, color='red', label='Giá trị dự đoán')
                if prediction is not None and input_value is not None:
                    ax.scatter(input_value[0], prediction, color='green', s=100, label='Giá trị dự đoán cụ thể', zorder=5)
                ax.set_xlabel(input_names[0])
                ax.set_ylabel(input_names[1])
                ax.set_zlabel(ylabel)
                ax.set_title(title)
                ax.legend()
                plt.show()
            
    
        def plot_logistic_model(ax):
            # Trường hợp hồi quy Logistic với 1 biến độc lập
                x_range = np.linspace(X.min(), X.max(), 300)
                x_values = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 300)
                y_values = model.predict_proba(x_range.reshape(-1, 1))[:, 1]
                cm = confusion_matrix(y, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                plt.plot(x_values, y_values, label='Xác suất dự đoán')
                plt.scatter(X, y, color='red', label='Dữ liệu thực tế', alpha=0.5)
                if prediction is not None and input_value is not None:
                    plt.scatter(input_value[0], prediction, color='green', s=100, label='Giá trị dự đoán cụ thể', zorder=5)

                plt.xlabel('X')
                plt.ylabel('Xác suất')
                plt.legend()
                ax.set_xlabel(input_names[0]) 
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend()
                plt.show()
                disp.plot(ax=plt.subplot(), cmap=plt.cm.Blues)
                plt.title(f"Ma trận nhầm lẫn - {title}")
                plt.show()
                
        def plot_knn_model(ax):
            if len(input_names) == 1:
                    # Nếu chỉ có một biến độc lập, vẽ biểu đồ scatter plot và đường thẳng nối điểm dữ liệu
                    ax.scatter(X.iloc[:, 0], y, color='blue', label='Giá trị thực tế')
                    ax.scatter(X.iloc[:, 0], y_pred, color='red', label='Giá trị dự đoán')
                    for i in range(len(X)):
                        x_i = X.iloc[i, 0] if isinstance(X, pd.DataFrame) else X[i]
                        y_i = y.iloc[i] if isinstance(y, pd.Series) else y[i]
                        y_pred_i = y_pred[i]
                        ax.plot([x_i, x_i], [y_i, y_pred_i], color='red')
                    ax.set_xlabel(input_names[0]) 
                    # Nếu có nhiều hơn một biến độc lập, vẽ biểu đồ scatter plot và đường thẳng nối điểm dữ liệu
            else:
                ax.scatter(X, y, color='blue', label='Giá trị thực tế')
                ax.scatter(X, y_pred, color='red', label='Giá trị dự đoán')
                for i in range(len(X)):
                    ax.plot([X.iloc[i, j] for j in range(X.shape[1])], [y.iloc[i], y_pred[i]], color='red')
                ax.set_xlabel('Biến độc lập')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            plt.show()
        
        def plot_decision_tree(ax):
            plt.figure(figsize=(20, 10))
            plot_tree(model, filled=True, feature_names=input_names, class_names=np.unique(y).astype(str), rounded=True, fontsize=12)
            plt.show()
            
        if model and isinstance(model, LinearRegression):
            plot_linear_model(ax)
        elif model and isinstance(model, LogisticRegression):
            plot_logistic_model(ax)
        elif isinstance(model, KNeighborsRegressor):
            plot_knn_model(ax)
        elif isinstance(model, DecisionTreeClassifier):
            plot_decision_tree(ax)
            

    def handle_missing_values(self):
        # Kiểm tra nếu không có dữ liệu được tải từ file CSV
        if self.data is None:
            messagebox.showerror("Lỗi", "Không thể xử lý giá trị thiếu vì chưa load dữ liệu từ file CSV.")
            return

        # Tạo một cửa sổ mới khi bấm xử lý data missing value
        missing_values_window = tk.Toplevel(self.root)
        missing_values_window.title("Xử lý giá trị thiếu")
        missing_values_window.geometry("600x400")

        # Tạo danh sách cột có giá trị thiếu
        columns_with_missing_values = self.data.columns[self.data.isnull().any()].tolist()

        # Biến lưu trữ cột được chọn từ combobox
        selected_column_var = tk.StringVar()
        tk.Label(missing_values_window, text="Chọn cột có giá trị thiếu:").pack(pady=(0, 5))
        column_menu = ttk.Combobox(missing_values_window, textvariable=selected_column_var, values=columns_with_missing_values, state='readonly')
        column_menu.pack(pady=(0, 20))

        # Biến lưu trữ giá trị được nhập vào entry
        value_entry_var = tk.StringVar()
        tk.Label(missing_values_window, text="Hoặc nhập giá trị cụ thể:").pack(pady=(0, 5))
        value_entry = tk.Entry(missing_values_window, textvariable=value_entry_var)
        value_entry.pack(pady=(0, 20))

        # Hàm áp dụng giá trị vào giá trị thiếu
        def apply_value(specific_value=None):
            selected_column = selected_column_var.get()
            if specific_value is None:
                specific_value = value_entry_var.get()
            if selected_column and specific_value:
                  # Điền giá trị cụ thể vào giá trị thiếu trong cột đã chọn
                self.data[selected_column].fillna(specific_value, inplace=True)
                 # Hiển thị thông báo khi điền giá trị thành công 
                messagebox.showinfo("Thông báo", f"Giá trị Missing Value trong cột '{selected_column}' đã được điền bằng '{specific_value}'.")
                missing_values_window.destroy()
                 # Hiển thị lại danh sách biến và refresh data
                self.show_variables()
                self.refresh_data()

        # Hàm xóa dòng chứa giá trị thiếu
        def delete_row():
            selected_column = selected_column_var.get()
            if selected_column:
                 # Xóa các dòng chứa giá trị thiếu trong cột đã chọn
                self.data = self.data[self.data[selected_column].notnull()]
                messagebox.showinfo("Thông báo", f"Các dòng chứa giá trị Missing Value trong cột '{selected_column}' đã được xóa.")
                missing_values_window.destroy()
                self.show_variables()
                self.refresh_data()

        apply_button = tk.Button(missing_values_window, text="Áp dụng", width=15, command=apply_value)
        apply_button.pack(pady=(0, 20))

        tk.Label(missing_values_window, text="Chọn các giá trị để điền vào Missing Value").pack(pady=(0, 20))
        button_frame = tk.Frame(missing_values_window)
        button_frame.pack(side=tk.TOP, pady=(0, 10))

         # Button xử lý giao diện và điền giá trị trung bình, trung vị, mode và button xóa dòng
         
        mean_button = tk.Button(button_frame, text="Mean", width=15, command=lambda: apply_value(self.data[selected_column_var.get()].mean()))
        mean_button.pack(side=tk.LEFT, padx=15)

        median_button = tk.Button(button_frame, text="Median", width=15, command=lambda: apply_value(self.data[selected_column_var.get()].median()))
        median_button.pack(side=tk.LEFT, padx=15)

        mode_button = tk.Button(button_frame, text="Mode", width=15, command=lambda: apply_value(self.data[selected_column_var.get()].mode().iloc[0]))
        mode_button.pack(side=tk.LEFT, padx=15)

        delete_row_button = tk.Button(missing_values_window, text="Xóa dòng", width=15, command=delete_row)
        delete_row_button.pack(pady=(10, 0))

         # Chạy vòng lặp chính của cửa sổ xử lý giá trị thiếu
        missing_values_window.mainloop()


    # Hàm convert data
    def open_convert_window(self):
        new_window = tk.Toplevel(self.root)
        new_window.title("Xử lý Convert Data")
        new_window.geometry("400x300")

        tk.Label(new_window, text="Chọn cột để chuyển đổi:").pack(pady=10)
        self.convert_var = tk.StringVar()
        self.columns_combobox = ttk.Combobox(new_window, textvariable=self.convert_var, values=[col for col in self.data.columns if self.data[col].dtype == 'object'])
        self.columns_combobox.pack(pady=10)

        # Xử lý sự kiện convert Data khi bấm button Áp dụng
        convert_button = tk.Button(new_window, text="Áp dụng", command=self.apply_label_encoding)
        convert_button.pack(pady=20)
    
    
    # Hàm áp dụng labelencoding để xử lý convert data
    def apply_label_encoding(self):
        selected_column = self.convert_var.get()
        if selected_column:
            le = LabelEncoder()
            self.data[selected_column] = le.fit_transform(self.data[selected_column])
            messagebox.showinfo("Thành công", f"Cột '{selected_column}' đã được chuyển đổi thành số.")
            self.show_variables()
            self.refresh_data()


    # Hàm xử lý chức năng visualize
    def visualize_relationship(self):
        if self.data is None:
            messagebox.showerror("Lỗi", "Vui lòng chọn dữ liệu trước khi vẽ biểu đồ.")
            return

        # Hàm hiển thị cửa sổ chức năng Visualize
        viz_window = tk.Toplevel(self.root)
        viz_window.title("Visual Relationship")
        viz_window.geometry("1200x800")
        
        viz_window.protocol("WM_DELETE_WINDOW", lambda: self.clear_selected_variables(viz_window))

        variable_selection_frame = tk.Frame(viz_window)
        variable_selection_frame.pack(side=tk.LEFT, fill=tk.Y, padx=50, pady=10)

        # Frame xử lý hiển thị biểu đồ
        self.plot_frame = tk.Frame(viz_window)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame xử lý chọn biến
        tk.Label(variable_selection_frame, text="Chọn biến mục tiêu:").pack()
        target_var_visual = tk.StringVar()
        target_var_combobox_visual = ttk.Combobox(variable_selection_frame, textvariable=target_var_visual, values=list(self.data.columns), state='readonly')
        target_var_combobox_visual.pack(pady=(0, 10))

        add_target_button_visual = tk.Button(variable_selection_frame, text="Thêm biến mục tiêu", command=lambda: self.add_target_visual(target_var_visual.get()))
        add_target_button_visual.pack(pady=(0, 10))


        self.selected_target_listbox_visual = tk.Listbox(variable_selection_frame, height=4)
        self.selected_target_listbox_visual.pack(pady=(0, 10))

        remove_target_button_visual = tk.Button(variable_selection_frame, text="Xóa", command=self.remove_selected_target_visual)
        remove_target_button_visual.pack(pady=(0, 10))

        tk.Label(variable_selection_frame, text="Chọn biến độc lập:").pack()
        input_var_visual = tk.StringVar()
        input_var_combobox_visual = ttk.Combobox(variable_selection_frame, textvariable=input_var_visual, values=list(self.data.columns), state='readonly')
        input_var_combobox_visual.pack(pady=(0, 10))

        add_input_button_visual = tk.Button(variable_selection_frame, text="Thêm biến độc lập", command=lambda: self.add_input_visual(input_var_visual.get()))
        add_input_button_visual.pack(pady=(0, 10))


        self.selected_vars_listbox_visual = tk.Listbox(variable_selection_frame, height=6)
        self.selected_vars_listbox_visual.pack(pady=(0, 10))

        remove_input_button_visual = tk.Button(variable_selection_frame, text="Xóa", command=self.remove_selected_variable_visual)
        remove_input_button_visual.pack(pady=(0, 10))

       
        tk.Label(variable_selection_frame, text="Chọn biểu đồ cần vẽ:").pack()
        self.plot_type_combobox = ttk.Combobox(variable_selection_frame, state='readonly', width = 30)
        self.plot_type_combobox.pack(fill=tk.X, pady=(0, 10))

        # Hàm xử lý button thực hiện vẽ biểu đồ
        generate_button = tk.Button(variable_selection_frame, text="Thực hiện", command=self.handle_plot_selection)
        generate_button.pack(fill=tk.X, pady=(0, 10))
        
        
    
    # Hàm xử lý khi thêm biến mục tiêu và cập nhật trong combobox
    #Target Age
    def add_target_visual(self, target):
        if target and target not in self.selected_target_listbox_visual.get(0, tk.END):
            # Xử lý thêm biến đã chọn vào giao diện
            self.selected_target_listbox_visual.insert(tk.END, target)
        #       self.viz_selected_target_vars = [Age]
        #       self.viz_selected_input_vars = [Purchased]
            if target not in self.viz_selected_target_vars:
                self.viz_selected_target_vars.append(target)
            self.build_combobox_options()  
            

    # Hàm xử lý xóa khi xóa biến mục tiêu và cập nhật trong combobox
    def remove_selected_target_visual(self):
        selected_indices = self.selected_target_listbox_visual.curselection()
        if selected_indices:
            selected_index = selected_indices[0]
            selected_var = self.selected_target_listbox_visual.get(selected_index)
            if selected_var in self.viz_selected_target_vars:
                self.viz_selected_target_vars.remove(selected_var)
            self.selected_target_listbox_visual.delete(selected_index)
            self.build_combobox_options()  

    # Hàm xử lý thêm biến độc lập và cập nhật trong combobox
    def add_input_visual(self, input_var):
        if input_var and input_var not in self.selected_vars_listbox_visual.get(0, tk.END):
            self.selected_vars_listbox_visual.insert(tk.END, input_var)
            if input_var not in self.viz_selected_input_vars:
                self.viz_selected_input_vars.append(input_var)
            self.build_combobox_options()  

    # Hàm xử lý xóa khi xóa biến độc lập và cập nhật trong combobox
    def remove_selected_variable_visual(self):
        selected_indices = self.selected_vars_listbox_visual.curselection()
        if selected_indices:
            selected_index = selected_indices[0]
            selected_var = self.selected_vars_listbox_visual.get(selected_index)
            if selected_var in self.viz_selected_input_vars:
                self.viz_selected_input_vars.remove(selected_var)
            self.selected_vars_listbox_visual.delete(selected_index)
            self.build_combobox_options() 

    # Hàm xử lý thêm danh sách biểu đồ cần vẽ vào trong Combobox
    def build_combobox_options(self):
        # Histogram of Age, Purchased
        self.plot_options = [] 
        # Age viz_selected_target_vars
        numeric_targets = [var for var in self.viz_selected_target_vars if self.data[var].dtype in ['int64', 'float64']] 
        categorical_targets = [var for var in self.viz_selected_target_vars if self.data[var].dtype == 'object']
        # Purchased viz_selected_input_vars
        numeric_inputs = [var for var in self.viz_selected_input_vars if self.data[var].dtype in ['int64', 'float64']] 
        categorical_inputs = [var for var in self.viz_selected_input_vars if self.data[var].dtype == 'object'] 

        # Xử lý biểu đồ định lượng và phân loại 1 biến
        for var in numeric_inputs + numeric_targets:
            self.plot_options.append(f"Histogram of {var}")
        for var in categorical_inputs + categorical_targets:
            self.plot_options.append(f"Bar Chart of {var}")
            self.plot_options.append(f"Pie Chart of {var}")

        # Xử lý biểu đồ nhiều biến trường hợp
        if len(numeric_inputs + numeric_targets) >= 2:
            self.plot_options.append("Correlation Matrix between Numeric Variables")
            self.plot_options.append("Pair Plot between Numeric Variables")
        # Xử lý Scatter Plot cho trường hợp 1 biến mục tiêu 2 biến độc lập
        if len(numeric_inputs) == 2 and len(numeric_targets) == 1:
            var1, var2 = numeric_inputs
            target_var = numeric_targets[0]
            self.plot_options.append(f"3D Scatter Plot between {var1}, {var2} and {target_var}")

        # Xử lý các biểu đồ 1 biến phân loại 1 biến định lượng
        if categorical_inputs and (numeric_inputs or numeric_targets):
            for cat_var in categorical_inputs:
                for num_var in numeric_inputs + numeric_targets:
                    self.plot_options.append(f"Box Plot between {cat_var} and {num_var}")
                    self.plot_options.append(f"Violin Plot between {cat_var} and {num_var}")
           
        # if len(categorical_inputs) >= 2:
        #     self.plot_options.append("Count Plot between Categorical Variables")

        self.plot_type_combobox['values'] = self.plot_options # (Histogram, Matrix, Pair)
        self.plot_type_combobox.set('')

 

    # Nhận sự kiện trong combobox để xử lý vẽ biểu đồ tương ứng
    def handle_plot_selection(self):
        selected_plot = self.plot_type_combobox.get()
        if "Histogram of" in selected_plot:
            var = selected_plot.split(" of ")[1]
            self.draw_histogram(var)
        elif "Bar Chart of" in selected_plot:
            var = selected_plot.split(" of ")[1]
            self.draw_bar_chart(var)
        elif "Pie Chart of" in selected_plot:
            var = selected_plot.split(" of ")[1]
            self.draw_pie_chart(var)
        elif "Box Plot between" in selected_plot:
            _, parts = selected_plot.split(" between ")
            cat_var, num_var = parts.split(" and ")
            self.draw_box_plot(cat_var, num_var)
        elif "Violin Plot between" in selected_plot:
            _, parts = selected_plot.split(" between ")
            cat_var, num_var = parts.split(" and ")
            self.draw_violin_plot(cat_var, num_var)
        elif "Correlation Matrix" in selected_plot:
            self.draw_correlation_matrix()
        elif "Pair Plot" in selected_plot:
            self.draw_pair_plot()
        elif "3D Scatter Plot between" in selected_plot:
            components = selected_plot.split(" between ")[1].split(" and ")
            if len(components) == 2:
                var1, var2 = components[0].split(", ")
                target_var = components[1]
                self.draw_scatter_plot(var1.strip(), var2.strip(), target_var.strip())
                print(var1.strip())
                print(var2.strip())
                print(target_var.strip())
        elif "Count Plot" in selected_plot:
            self.draw_count_plot()
    
    # Hàm xử lý xóa biến đã chọn khi tắt dialog
    def clear_selected_variables(self, window):
        self.selected_target_listbox_visual.delete(0, tk.END)
        self.selected_vars_listbox_visual.delete(0, tk.END)
        self.viz_selected_target_vars.clear()
        self.viz_selected_input_vars.clear()
        window.destroy()
        
       
    
    # Hàm xử lý vẽ Scatter Plot
    def draw_scatter_plot(self, var1, var2, target_var):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        sns.scatterplot(x=self.data[var1], y=self.data[var2], hue=self.data[target_var], palette='viridis', ax=ax)
        ax.set_title(f"Scatter Plot between {var1} and {var2} by {target_var}")
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Hàm xử lý vẽ Histogram
    def draw_histogram(self, var):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(self.data[var], bins=30, alpha=0.7, color='blue')
        ax.set_title(f"Histogram of {var}")
        ax.set_xlabel(var)
        ax.set_ylabel('Count')
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Hàm xử lý vẽ Box Plot
    def draw_box_plot(self, input_var, target_var):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        sns.boxplot(x=self.data[input_var], y=self.data[target_var], ax=ax)
        ax.set_title(f"Box Plot of {input_var} vs {target_var}")
        ax.set_xlabel(input_var)
        ax.set_ylabel(target_var)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Hàm xử lý vẽ Bar Chart
    def draw_bar_chart(self, var):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        data_count = self.data[var].value_counts()
        data_count.plot(kind='bar', ax=ax)
        ax.set_title(f'Bar Chart of {var}')
        ax.set_xlabel(var)
        ax.set_ylabel('Counts')
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Hàm xử lý vẽ biểu đồ Pie Chart
    def draw_pie_chart(self, var):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        self.data[var].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title(f'Pie Chart of {var}')
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    
    # Hàm xử lý vẽ biểu đồ Violin
    def draw_violin_plot(self, cat_var, num_var):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        sns.violinplot(x=self.data[cat_var], y=self.data[num_var], ax=ax)
        ax.set_title(f"Violin Plot between {cat_var} and {num_var}")
        ax.set_xlabel(cat_var)
        ax.set_ylabel(num_var)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Hàm xử lý vẽ biểu đồ Correlation Matrix
    def draw_correlation_matrix(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        numeric_vars = [var for var in self.viz_selected_input_vars + self.viz_selected_target_vars if self.data[var].dtype in ['int64', 'float64']]
        if numeric_vars:
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            corr_matrix = self.data[numeric_vars].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Matrix")
            canvas = FigureCanvasTkAgg(fig, self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Test
    def draw_count_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        sns.countplot(data=self.data, x=self.viz_selected_input_vars[0], hue=self.viz_selected_input_vars[1], ax=ax)
        ax.set_title("Count Plot between all Categorical Variables")
        ax.set_xlabel(self.viz_selected_input_vars[0])
        ax.set_ylabel('Count')
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Hàm xử lý vẽ biểu đồ Pair Plot
    def draw_pair_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        input_vars = self.viz_selected_input_vars + self.viz_selected_target_vars
        sns_plot = sns.pairplot(self.data[input_vars])
        sns_plot.fig.suptitle("Pair Plot between all Quantitative Variables", y=1.02)
        canvas = FigureCanvasTkAgg(sns_plot.fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    

if __name__ == "__main__":
    root = tk.Tk()
    app = RegressionApp(root)
    root.mainloop()


