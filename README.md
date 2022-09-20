# News Categorizer
This project uses some news sentences within their categories as train data to categorize test data, which are some other news sentences, in recognized news categories.<br>
It implements an NLP system uses unigram and bigram modeling with back-off smoothing to classify news in different news categories.

At first, train text file is read and unigram and bigram models is saved.<br>
Then sentences of test file is examined with train dataset and its catagory is predicted.<br>
At last, precision, recall and f-measure parameters is printed.

In this project back-off smoothing is implemented; that works with conditional probability (by Bayes law) and two coefficiant: landa and 1 - landa.

Train and test files are news sentences in Persian and the category of each sentence is written at the first of that sentence before '@' characters. One sentense is inserted below. It is an economic news:

اقتصاد@@@@@@@@@@ بزرگترین واحد تولید پارازایلین خاورمیانه به زودی افتتاح می شود گروه اقتصادی بزرگترین واحد تولید پارازایلین خاورمیانه با هدف صدور سالانه تا میلیون دلار پارازایلین توسط مقامات عالیرتبه در بندر امام افتتاح خواهد شد محمدرضا رضایی مدیر طرح پارازایلین بندرامام روز سه شنبه در جمع خبرنگاران گفت ظرفیت این واحد هزار تن در سال است و افزود ظرفیت تولید پارازایلین در واحد پارازایلین بندرامام برابرمیزان مصرف کشور است تولیدات سال اول این واحد برای بازپرداخت سرمایه گذاری صادر خواهد شد در حال حاضر واحد پارازایلین مجتمع پتروشیمی اصفهان با تولید هزارتن در سال احتیاج صنایع کشور به این ماده را تامین می کند وی افزود این طرح در چارچوب دومین برنامه دوم توسعه اقتصادی کشور وبا هزینه میلیون دلار و میلیارد ریال احداث شد در برآوردهای اولیه هزینه ارزی ساخت این واحد میلیون دلار و هزینه ریالی آن میلیاردریال پیش بینی شده بود وی کنترل هزینه راه اندازی را یکی از دستاوردهای مهم طرح پارازایلین بندر امام خواند رضایی گفت استفاده از نیروهای داخلی در تمام مراحل نصب و ساختمان از جمله دیگر دستاوردهای این طرح می باشد وی افزود برج متری جداسازی مخلوط زایلین ها بلندترین برج تقطیرمجتمع پتروشیمی بندر امام و منطقه خاورمیانه است که تمام بخش های آن درایران ساخته و نصب شده است از هزار تن تجهیزات اصلی استفاده شده در این طرح حدود هزار و تن ساخت داخل و هزار و تن از خارج خریداری شده است فناوری ساخت این واحد از موسسه تحقیقات نفت فرانسه دریافت شده و به لحاظ فناوری پیشرفته ترین نوع در خاورمیانه است مدیر طرح پارازایلین بندرامام اظهار داشت با راه اندازی واحد تولیدپارازایلین در مجتمع پتروشیمی بندرامام بیش از هزار فرصت شغلی درواحدهای پتروشیمی و واحدهای پایین دستی آن ایجاد می شود 
