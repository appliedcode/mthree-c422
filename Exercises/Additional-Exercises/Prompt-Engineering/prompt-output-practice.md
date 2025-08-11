## **📝 Text Generation - Additional Exercises**

### **Exercise A1: Explanations (Beginner)**

```python
# Try these prompts and observe how explanations change:

prompts = [
    "Explain photosynthesis to a kindergarten student using only simple words.",
    "Explain photosynthesis to a high school biology student.",
    "Explain photosynthesis to someone who has never heard of plants.",
    "Explain photosynthesis as if you're a plant talking to another plant."
]

for prompt in prompts:
    print(f"PROMPT: {prompt}")
    print(f"RESPONSE: {gpt_request(prompt)}")
    print("-" * 60)
```


### **Exercise A2: Essay Writing (Intermediate)**

```python
# Practice different essay types:

essay_prompts = [
    "Write a 200-word persuasive essay on why homework should be banned in elementary schools.",
    "Write a 150-word compare-and-contrast essay on books vs. movies.",
    "Write a 250-word narrative essay about a day when gravity stopped working.",
    "Write a 200-word argumentative essay on whether social media does more harm than good."
]

# Your task: Try each prompt and note differences in structure, tone, and style
```


### **Exercise A3: Email Communication (Advanced)**

```python
# Practice professional communication:

email_scenarios = [
    "Write an email to your manager explaining why you need to work from home for a week.",
    "Write a follow-up email after a job interview, thanking the interviewer.",
    "Write an email to customer service complaining about a defective product (keep it professional).",
    "Write an email to a colleague asking them to cover your shift, but you've asked before.",
    "Write an email declining a social invitation without hurting feelings."
]

# Challenge: Pay attention to tone, formality level, and structure
```


### **Exercise A4: Creative Writing**

```python
# Experiment with creative prompts:

creative_prompts = [
    "Write a product description for a time machine that makes everything sound normal and boring.",
    "Write a news report about penguins learning to fly, written in a serious journalistic tone.",
    "Write a restaurant review for a place that serves food from the future.",
    "Write instructions for teaching your pet goldfish to play chess."
]
```


***

## **💻 Code Generation - Additional Exercises**

### **Exercise B1: Basic Code Generation**

```python
# Try generating different types of functions:

code_prompts = [
    "Write a Python function that checks if a number is prime. Include docstring and examples.",
    "Write a Python function that converts temperature from Celsius to Fahrenheit and Kelvin.",
    "Write a Python class for a simple bank account with deposit, withdraw, and balance methods.",
    "Write Python code to find the longest word in a sentence.",
    "Write a Python function that generates the Fibonacci sequence up to n numbers."
]

# Practice: Run the generated code to see if it works correctly!
```


### **Exercise B2: Code Refactoring Practice**

```python
# Refactor these poorly written code snippets:

messy_code_1 = """
def calc(x,y,z):
    if z=='add':
        return x+y
    elif z=='sub':
        return x-y
    elif z=='mul':
        return x*y
    elif z=='div':
        return x/y
"""

messy_code_2 = """
data = [1,2,3,4,5,6,7,8,9,10]
result = []
for i in data:
    if i%2==0:
        result.append(i*i)
print(result)
"""

messy_code_3 = """
def process_data(lst):
    new_lst = []
    for item in lst:
        if item > 0:
            if item < 100:
                if item % 2 == 0:
                    new_lst.append(item * 2)
    return new_lst
"""

# Your task: Create prompts to refactor each code snippet for better readability, efficiency, and style
```


### **Exercise B3: Debugging Challenge**

```python
# Fix these buggy code snippets:

buggy_code_1 = """
def average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # What if numbers is empty?

print(average([]))
"""

buggy_code_2 = """
def count_vowels(text):
    vowels = 'aeiou'
    count = 0
    for char in text:
        if char in vowels:  # What about uppercase?
            count += 1
    return count

print(count_vowels("Hello World"))
"""

buggy_code_3 = """
def factorial(n):
    if n == 1:  # What about n=0?
        return 1
    return n * factorial(n-1)

print(factorial(0))
print(factorial(-5))  # What happens here?
"""

# Practice: Create debugging prompts that explain the issues and provide fixes
```


### **Exercise B4: Algorithm Implementation**

```python
# Advanced coding challenges:

algorithm_prompts = [
    "Implement binary search in Python with detailed comments explaining each step.",
    "Write Python code for a simple text-based tic-tac-toe game.",
    "Implement a Python function to detect if a string is a valid palindrome (ignore spaces and punctuation).",
    "Write Python code to implement a basic stack data structure with push, pop, and peek methods.",
    "Create a Python function that sorts a list of dictionaries by multiple keys."
]
```


***

## **📄 Summarization - Additional Exercises**

### **Exercise C1: News Article Summarization**

```python
# Practice with different article types:

tech_article = """
Apple announced its latest iPhone model yesterday, featuring significant improvements in battery life, camera quality, and processing speed. The new A18 chip delivers 40% better performance than the previous generation while using 30% less power. The camera system now includes advanced AI-powered photography features, including real-time object recognition and automatic scene optimization. The device will be available in four colors and three storage capacities, with prices starting at $899. Pre-orders begin next Friday, with general availability expected in early September. Early reviews from tech journalists have been largely positive, praising the battery improvements and camera quality, though some criticized the incremental nature of the upgrades and the high price point.
"""

science_article = """
Researchers at MIT have developed a new type of solar panel that can generate electricity even in low-light conditions. The breakthrough technology uses a novel combination of materials called perovskites, which can capture different wavelengths of light more efficiently than traditional silicon panels. Laboratory tests show these panels can produce 60% more electricity than conventional solar panels in cloudy conditions and can even generate small amounts of power from moonlight and artificial lighting. The research team believes this technology could revolutionize solar energy adoption in regions with limited sunlight. However, the technology is still in early development stages, and researchers estimate it will take 5-7 years before commercial applications become available. The main challenges remaining include improving the long-term stability of perovskite materials and reducing manufacturing costs.
"""

# Your tasks:
# 1. Create extractive summaries (pulling key sentences directly)
# 2. Create abstractive summaries (rewriting in your own words)
# 3. Create different length summaries (1 sentence, 3 sentences, full paragraph)
```


### **Exercise C2: Academic Paper Summarization**

```python
abstract = """
This study examines the relationship between social media usage and sleep quality among college students. We surveyed 500 undergraduate students about their daily social media consumption and sleep patterns over a six-week period. Participants who spent more than 3 hours daily on social media platforms showed significantly worse sleep quality scores, with 73% reporting difficulty falling asleep and 68% experiencing frequent nighttime awakenings. The study also found that social media use within 2 hours of bedtime was particularly detrimental to sleep quality. Interestingly, the type of social media activity mattered: passive consumption (scrolling, watching) had stronger negative effects than active engagement (posting, commenting). These findings suggest that limiting social media use, especially before bedtime, could improve sleep quality among college students. Future research should investigate specific intervention strategies and examine whether these effects persist across different age groups.
"""

# Practice different summary styles:
# 1. Executive summary for university administrators
# 2. Key findings summary for students
# 3. Methodology summary for other researchers
# 4. One-sentence summary for a headline
```


### **Exercise C3: Multi-Document Summarization**

```python
# Combine information from multiple sources:

source_1 = "Electric vehicles are becoming more popular due to environmental concerns and government incentives. Sales increased by 75% last year."

source_2 = "The main challenges for electric vehicle adoption include limited charging infrastructure and higher upfront costs compared to gasoline cars."

source_3 = "Battery technology improvements have extended electric vehicle range to over 300 miles per charge, addressing previous consumer concerns."

source_4 = "Major automakers have committed to transitioning their entire fleets to electric vehicles by 2035, with some targeting even earlier dates."

# Challenge: Create a comprehensive summary that synthesizes information from all sources
```


***

## **🏷️ Classification/Labeling - Additional Exercises**

### **Exercise D1: Sentiment Analysis Variations**

```python
# Practice with different types of content:

product_reviews = [
    "This phone case is okay, nothing special but it does protect my phone adequately.",
    "Absolutely love this case! Perfect fit, great protection, and looks amazing!",
    "Terrible quality. Broke after just one week of normal use. Complete waste of money.",
    "Good value for the price. Not premium quality but gets the job done.",
    "The case is beautiful but makes the phone very bulky. Mixed feelings about this purchase."
]

restaurant_reviews = [
    "Food was cold when it arrived and the service was incredibly slow. Won't be back.",
    "Amazing flavors and the staff was so friendly! Definitely my new favorite restaurant.",
    "The atmosphere is nice but the food is overpriced for the portion sizes.",
    "Decent food, nothing extraordinary but good for a quick meal.",
    "Best pasta I've ever had! The chef really knows what they're doing."
]

# Practice: Try different classification schemes:
# - 3-point scale (Positive/Neutral/Negative)
# - 5-star rating system
# - Emotion classification (Happy/Sad/Angry/Neutral/Excited)
```


### **Exercise D2: Topic Classification**

```python
# Classify news headlines by category:

headlines = [
    "New AI breakthrough could revolutionize medical diagnosis",
    "Stock market reaches record high as inflation concerns ease",
    "Local high school wins state championship in dramatic overtime",
    "Climate change report shows urgent need for action",
    "Celebrity couple announces surprise wedding in private ceremony",
    "New vaccine shows 95% effectiveness in clinical trials",
    "Tech giant announces major layoffs affecting thousands",
    "Archaeological discovery sheds light on ancient civilization"
]

# Categories: Technology, Finance, Sports, Environment, Entertainment, Health, Business, Science
```


### **Exercise D3: Content Moderation**

```python
# Practice content safety classification:

social_media_posts = [
    "Just had the best day at the beach with friends! 🌊☀️",
    "I disagree with the new policy, but I respect different viewpoints.",
    "This politician is the worst thing that ever happened to our country!!!",
    "Check out my new art project - spent weeks working on it!",
    "Can't believe how stupid some people are. They shouldn't be allowed to vote.",
    "Having trouble with my homework, can anyone help me understand calculus?",
    "Everyone from that country is lazy and worthless.",
    "Love seeing all the support for the charity event! Great community spirit."
]

# Classify as: Appropriate, Needs Review, or Inappropriate
# Also identify: Helpful, Neutral, Toxic, Spam
```


### **Exercise D4: Intent Classification**

```python
# Customer service intent recognition:

customer_messages = [
    "I can't log into my account, it says my password is wrong",
    "When will my order arrive? I placed it 5 days ago",
    "I want to return this item, it doesn't fit properly",
    "How do I cancel my subscription?",
    "Your website is down, I can't access anything",
    "I was charged twice for the same purchase",
    "Do you have this product in a different color?",
    "I love this product! Just wanted to say thanks!"
]

# Intents: Technical Support, Order Status, Returns, Account Management, 
# Bug Report, Billing Issue, Product Inquiry, Compliment
```


***

## **🗂️ Tables/JSON Generation - Additional Exercises**

### **Exercise E1: Data Structure Creation**

```python
# Practice generating different data formats:

data_requests = [
    "Create a JSON object representing a recipe with ingredients, instructions, prep time, and difficulty level",
    "Generate a table comparing iPhone, Samsung Galaxy, and Google Pixel phones across price, camera, battery, and storage",
    "Create a JSON array of 5 books with title, author, publication year, genre, and rating",
    "Make a table showing the top 5 programming languages with their use cases, difficulty, and average salary",
    "Generate JSON data for a simple e-commerce product catalog with 3 items"
]
```


### **Exercise E2: Complex Data Structures**

```python
# Advanced JSON challenges:

complex_requests = [
    """Create a JSON object for a university course catalog entry including:
    - Course information (code, title, credits, description)
    - Prerequisites (array of course codes)
    - Schedule options (multiple sections with days, times, instructor)
    - Assessment methods (exams, assignments with weights)""",
    
    """Generate a JSON structure for a restaurant menu including:
    - Categories (appetizers, mains, desserts)
    - Each item with name, description, price, dietary restrictions
    - Daily specials with dates
    - Nutritional information where applicable""",
    
    """Create a JSON object for a movie database entry with:
    - Basic info (title, year, genre, runtime)
    - Cast and crew (with roles)
    - Ratings from different sources
    - Box office data
    - Streaming availability"""
]
```


### **Exercise E3: CSV/Table Generation**

```python
# Practice creating different table formats:

table_requests = [
    "Create a study schedule table for a student taking 5 courses, showing daily time slots and subjects",
    "Generate a budget tracking table with categories, planned vs actual spending, and variance",
    "Make a fitness tracker table showing weekly workout plans with exercises, sets, reps, and rest periods",
    "Create an inventory table for a small bookstore with ISBN, title, author, quantity, and price",
    "Generate a travel itinerary table with dates, locations, activities, and estimated costs"
]
```


### **Exercise E4: Data Validation**

```python
# Generate data with specific constraints:

validation_challenges = [
    "Create JSON for 10 employees ensuring all email addresses are valid format and phone numbers follow US format",
    "Generate a table of 20 products where prices are between $10-$1000 and all SKUs follow format ABC-1234",
    "Create JSON for a class roster where student IDs are 8 digits and GPAs are between 0.0-4.0",
    "Make a table of appointments where all dates are in the future and times are in 30-minute intervals"
]
```


***

## **⚖️ Evaluation/Comparison - Additional Exercises**

### **Exercise F1: Algorithm Comparison**

```python
# Compare different approaches:

sorting_algorithms = {
    "bubble_sort": "Simple but inefficient algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in wrong order.",
    "quicksort": "Efficient divide-and-conquer algorithm that picks a 'pivot' element and partitions array around the pivot.",
    "merge_sort": "Stable, divide-and-conquer algorithm that divides the array into halves, sorts them, then merges the sorted halves."
}

# Task: Create prompts to compare these algorithms across:
# - Time complexity
# - Space complexity  
# - Stability
# - Best use cases
# - Implementation difficulty
```


### **Exercise F2: Product Evaluation**

```python
# Compare similar products or services:

comparison_scenarios = [
    "Compare Netflix, Disney+, and HBO Max for a family with young children and teenagers",
    "Evaluate MacBook Air vs Dell XPS 13 vs ThinkPad X1 Carbon for a college student",
    "Compare Tesla Model 3, BMW i4, and Audi e-tron GT for an environmentally conscious professional",
    "Evaluate Google Workspace vs Microsoft 365 vs Apple iWork for a small business",
    "Compare Spotify, Apple Music, and YouTube Music for different types of music listeners"
]

# Focus on: Features, pricing, user experience, pros/cons, recommendations
```


### **Exercise F3: Writing Quality Assessment**

```python
# Evaluate different versions of the same content:

email_version_1 = """
Hi,
I need the report by tomorrow. Please send it ASAP.
Thanks.
"""

email_version_2 = """
Dear [Name],
I hope this email finds you well. I wanted to follow up on the quarterly report we discussed last week. Due to an unexpected client meeting that was just scheduled, I would greatly appreciate if you could prioritize completing the report by tomorrow morning if possible. Please let me know if you need any additional resources or if this timeline presents any challenges.
Thank you for your understanding and flexibility.
Best regards,
[Your name]
"""

email_version_3 = """
Hello [Name],
I hope you're having a good week! I wanted to check in about the quarterly report we discussed. Due to a client presentation that came up suddenly, would it be possible to have the report completed by tomorrow? I know it's short notice, so please let me know if you need anything to help meet this deadline.
Thanks so much for your help!
[Your name]
"""

# Evaluation criteria: Tone, professionalism, clarity, politeness, effectiveness
```


### **Exercise F4: Code Quality Assessment**

```python
# Compare different implementations:

function_v1 = """
def find_max(lst):
    max_val = lst[0]
    for i in range(1, len(lst)):
        if lst[i] > max_val:
            max_val = lst[i]
    return max_val
"""

function_v2 = """
def find_max(lst):
    return max(lst)
"""

function_v3 = """
def find_max(lst):
    if not lst:
        raise ValueError("Cannot find max of empty list")
    
    max_val = lst[0]
    for item in lst[1:]:
        if item > max_val:
            max_val = item
    return max_val
"""

# Compare across: Readability, efficiency, error handling, pythonic style, robustness
```


***

## **🎯 Challenge Exercises (Mixed Skills)**

### **Challenge 1: Multi-Modal Content Creation**

```python
# Create content that combines multiple output types:

challenge_1 = """
Create a mini-lesson on binary search that includes:
1. A simple explanation for beginners
2. Python code implementation with comments
3. JSON data showing step-by-step example
4. Comparison table: Binary vs Linear search
5. Classification of when to use each method
"""
```


### **Challenge 2: Content Adaptation**

```python
# Adapt the same information for different audiences:

challenge_2 = """
Take the concept of machine learning and create:
1. An explanation for a 10-year-old
2. An explanation for a business executive
3. An explanation for a computer science student
4. A JSON summary of key points
5. A comparison with human learning
"""
```


### **Challenge 3: Problem-Solution Framework**

```python
# Address a real-world scenario:

challenge_3 = """
A small restaurant wants to implement online ordering. Create:
1. A requirements analysis (what they need)
2. Comparison table of different solutions
3. JSON structure for their menu system  
4. Sample code for order processing
5. Evaluation criteria for choosing a platform
"""
```


***

## **📊 Self-Assessment Rubric**

Rate your outputs on a scale of 1-5:

### **Content Quality**

- **5**: Comprehensive, accurate, well-structured
- **3**: Good but missing some details
- **1**: Basic or inaccurate information


### **Prompt Effectiveness**

- **5**: Clear, specific, gets exactly what you wanted
- **3**: Generally good with minor issues
- **1**: Vague or produces unexpected results


### **Technical Accuracy**

- **5**: Code runs perfectly, data is valid
- **3**: Minor issues that are easily fixed
- **1**: Major errors or doesn't work

***

## **💡 Tips for Success**

1. **Start Simple**: Begin with basic prompts and gradually add complexity
2. **Iterate**: If the output isn't quite right, refine your prompt
3. **Test Code**: Always run generated code to verify it works
4. **Validate Data**: Check that JSON is properly formatted
5. **Be Specific**: The more specific your prompt, the better the output
6. **Experiment**: Try different phrasings to see how they affect results

***

**🏆 Mastery Goal**: Complete at least 3 exercises from each category and 1 challenge exercise. Document what you learned about effective prompt engineering!

