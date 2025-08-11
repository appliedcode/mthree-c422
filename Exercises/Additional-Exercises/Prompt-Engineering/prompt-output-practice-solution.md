## **📝 Text Generation Solutions**

### **Exercise A1: Explanations - Sample Solutions**

```python
# Solution 1: Age-appropriate explanations
prompt_kindergarten = "Explain photosynthesis to a kindergarten student using only simple words."
# Expected approach: Use simple analogies, avoid technical terms
sample_response_kindergarten = """
Plants are like little food makers! They take in sunshine through their green leaves, 
drink water through their roots, and breathe in air. Then they mix it all together 
to make their own food, just like when you mix ingredients to make cookies! 
And the best part? They give us fresh air to breathe!
"""

prompt_high_school = "Explain photosynthesis to a high school biology student."
# Expected approach: Include scientific terms, process steps, chemical equation
sample_response_high_school = """
Photosynthesis is the process where plants convert light energy into chemical energy. 
It occurs in two main stages: the light-dependent reactions in the thylakoids and 
the light-independent reactions (Calvin cycle) in the stroma of chloroplasts. 
The overall equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂
This process is crucial for life on Earth as it produces oxygen and glucose.
"""

print("✅ Key Learning: Adjust vocabulary and complexity based on audience level")
```


### **Exercise A2: Essay Writing - Sample Solutions**

```python
# Solution for persuasive essay
prompt_persuasive = "Write a 200-word persuasive essay on why homework should be banned in elementary schools."
# Good prompt includes: word count, essay type, clear position, target audience

sample_solution_persuasive = """
Elementary school students should not be burdened with homework after their long school days. 
Research shows that homework in elementary grades provides no significant academic benefit while 
causing unnecessary stress for both children and families.

Young children learn best through play, exploration, and hands-on experiences, not through 
repetitive worksheets at home. After spending 6-7 hours in structured learning environments, 
children need time to decompress, be creative, and develop social skills through unstructured play.

Furthermore, homework creates inequality among students. Not all families can provide quiet study 
spaces, parental help, or necessary supplies. This disadvantages students from working families 
or those without educational resources, widening the achievement gap rather than closing it.

Instead of homework, elementary students should spend their evenings reading for pleasure, 
engaging in physical activities, and bonding with family members. These activities contribute 
more meaningfully to their overall development than worksheet completion.

Schools should focus on making classroom time more effective rather than extending learning 
into home hours. By eliminating elementary homework, we can reduce family stress, promote 
equity, and allow children to develop into well-rounded individuals.
"""

print("✅ Key Elements: Clear thesis, supporting evidence, counterargument consideration, strong conclusion")
```


### **Exercise A3: Email Communication - Sample Solutions**

```python
# Solution for work-from-home request
prompt_wfh = "Write an email to your manager explaining why you need to work from home for a week."
# Good approach: Professional tone, clear reason, show consideration for work impact

sample_solution_wfh = """
Subject: Request to Work from Home - [Your Name]

Dear [Manager's Name],

I hope this email finds you well. I am writing to request permission to work from home 
for the week of [dates] due to a family situation that requires my presence at home.

My [family member] is recovering from a medical procedure and needs assistance during 
the day. Working from home would allow me to provide necessary support while maintaining 
my work responsibilities and productivity.

I have reviewed my schedule and confirmed that all my meetings that week can be conducted 
virtually. I will be available during regular business hours and will ensure all deadlines 
are met as usual. I plan to check in daily via our team chat and provide updates on my 
project progress.

Please let me know if you need any additional information or if there are any concerns 
about this arrangement. I appreciate your understanding and flexibility during this time.

Thank you for your consideration.

Best regards,
[Your Name]
[Your Contact Information]
"""

print("✅ Key Elements: Clear subject line, reason without oversharing, work impact addressed, professional tone")
```


***

## **💻 Code Generation Solutions**

### **Exercise B1: Basic Code Generation - Sample Solutions**

```python
# Solution for prime number checker
prompt_prime = "Write a Python function that checks if a number is prime. Include docstring and examples."

sample_solution_prime = '''
def is_prime(n):
    """
    Check if a number is prime.
    
    Args:
        n (int): The number to check
        
    Returns:
        bool: True if the number is prime, False otherwise
        
    Examples:
        >>> is_prime(7)
        True
        >>> is_prime(10)
        False
        >>> is_prime(2)
        True
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to square root
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Test the function
test_numbers = [2, 3, 4, 17, 25, 29]
for num in test_numbers:
    print(f"{num} is prime: {is_prime(num)}")
'''

print("✅ Key Features: Proper docstring, edge cases handled, efficient algorithm, examples included")
```


### **Exercise B2: Code Refactoring - Sample Solutions**

```python
# Original messy code
messy_code = """
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

# Solution prompt
refactoring_prompt = f"""
Refactor this Python code to improve:
1. Function and variable names
2. Add type hints and docstring
3. Add input validation
4. Handle division by zero
5. Use more pythonic approaches

Original code:
{messy_code}
"""

# Expected refactored solution
sample_refactored_solution = '''
from typing import Union

def calculate(first_number: float, second_number: float, operation: str) -> Union[float, str]:
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        first_number: The first operand
        second_number: The second operand  
        operation: The operation to perform ('add', 'subtract', 'multiply', 'divide')
        
    Returns:
        The result of the calculation or error message
        
    Raises:
        ValueError: If operation is not supported
    """
    # Input validation
    valid_operations = {'add', 'subtract', 'multiply', 'divide'}
    if operation.lower() not in valid_operations:
        raise ValueError(f"Operation must be one of: {valid_operations}")
    
    operation = operation.lower()
    
    # Perform calculation
    if operation == 'add':
        return first_number + second_number
    elif operation == 'subtract':
        return first_number - second_number
    elif operation == 'multiply':
        return first_number * second_number
    elif operation == 'divide':
        if second_number == 0:
            return "Error: Cannot divide by zero"
        return first_number / second_number

# Alternative: Using dictionary dispatch (more pythonic)
def calculate_v2(first_number: float, second_number: float, operation: str) -> Union[float, str]:
    """More pythonic version using dictionary dispatch."""
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else "Error: Cannot divide by zero"
    }
    
    if operation.lower() not in operations:
        raise ValueError(f"Operation must be one of: {list(operations.keys())}")
        
    return operations[operation.lower()](first_number, second_number)
'''

print("✅ Improvements: Better names, type hints, error handling, documentation, pythonic style")
```


### **Exercise B3: Debugging - Sample Solutions**

```python
# Buggy factorial function
buggy_factorial = """
def factorial(n):
    if n == 0:
        return 0  # BUG: Should return 1, not 0
    else:
        return n * factorial(n-1)
"""

# Solution prompt
debug_prompt = f"""
Find and fix all bugs in this factorial function. Explain what's wrong and provide the corrected version:

{buggy_factorial}

Also add input validation for negative numbers.
"""

# Expected debugging solution
sample_debug_solution = '''
# BUGS IDENTIFIED:
# 1. factorial(0) should return 1, not 0 (by definition, 0! = 1)
# 2. No input validation for negative numbers
# 3. No handling for non-integer inputs

def factorial(n):
    """
    Calculate the factorial of a non-negative integer.
    
    Args:
        n (int): Non-negative integer
        
    Returns:
        int: Factorial of n
        
    Raises:
        ValueError: If n is negative or not an integer
    """
    # Input validation
    if not isinstance(n, int):
        raise ValueError("Input must be an integer")
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    # Base cases
    if n == 0 or n == 1:
        return 1
    
    # Recursive case
    return n * factorial(n - 1)

# Alternative iterative solution (more efficient for large numbers)
def factorial_iterative(n):
    """Iterative version - more memory efficient."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer")
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Test cases
test_cases = [0, 1, 5, 10]
print("Recursive version:")
for test in test_cases:
    print(f"{test}! = {factorial(test)}")

print("\nIterative version:")
for test in test_cases:
    print(f"{test}! = {factorial_iterative(test)}")
'''

print("✅ Key Fixes: Correct base case, input validation, clear error messages, alternative approach")
```


***

## **📄 Summarization Solutions**

### **Exercise C1: News Article Summarization - Sample Solutions**

```python
# Tech article about new iPhone
tech_article = """
Apple announced its latest iPhone model yesterday, featuring significant improvements in battery life, camera quality, and processing speed...
"""

# Extractive summary solution
extractive_prompt = f"Create an extractive summary by selecting the 2 most important sentences from this article:\n{tech_article}"

sample_extractive_solution = """
Apple announced its latest iPhone model yesterday, featuring significant improvements in battery life, camera quality, and processing speed. The new A18 chip delivers 40% better performance than the previous generation while using 30% less power.
"""

# Abstractive summary solution  
abstractive_prompt = f"Create a concise abstractive summary (2-3 sentences) of this article in your own words:\n{tech_article}"

sample_abstractive_solution = """
Apple has unveiled its newest iPhone with enhanced performance and efficiency features, including a more powerful A18 processor and improved photography capabilities. While early reviews praise the technical improvements, some critics question whether the incremental upgrades justify the $899 starting price point.
"""

# Different length summaries
one_sentence_prompt = f"Summarize this article in exactly one sentence:\n{tech_article}"
sample_one_sentence = "Apple's new iPhone features improved battery life, camera quality, and processing speed with the new A18 chip, starting at $899."

print("✅ Key Techniques: Extractive vs abstractive, length control, audience consideration")
```


### **Exercise C2: Academic Paper Summarization - Sample Solutions**

```python
# Academic abstract about social media and sleep
abstract = """
This study examines the relationship between social media usage and sleep quality among college students...
"""

# Executive summary for administrators
admin_prompt = f"Create an executive summary of this research for university administrators focused on actionable insights:\n{abstract}"

sample_admin_summary = """
EXECUTIVE SUMMARY: Social Media Impact on Student Sleep Quality

KEY FINDINGS:
- 73% of heavy social media users (3+ hours daily) report sleep difficulties
- Bedtime social media use significantly disrupts sleep patterns
- Passive consumption is more harmful than active engagement

RECOMMENDED ACTIONS:
- Implement digital wellness workshops during orientation
- Create campus-wide awareness campaign about healthy technology use
- Consider establishing "device-free" zones in residence halls
- Develop partnerships with counseling services for sleep hygiene education

IMPACT: Addressing these issues could improve student academic performance and mental health outcomes.
"""

# Key findings for students
student_prompt = f"Summarize the key findings from this study in simple terms for college students:\n{abstract}"

sample_student_summary = """
New research shows that spending more than 3 hours a day on social media can seriously mess with your sleep. 
Here's what they found:
• 7 out of 10 heavy social media users have trouble falling asleep
• Using social media within 2 hours of bedtime is particularly bad for sleep
• Just scrolling and watching (passive use) is worse for sleep than actually posting and interacting
• Better sleep might be as simple as putting your phone down earlier in the evening

Bottom line: If you're having sleep issues, try limiting social media before bed!
"""

print("✅ Audience Adaptation: Different focus, tone, and recommendations for each audience")
```


***

## **🏷️ Classification/Labeling Solutions**

### **Exercise D1: Sentiment Analysis - Sample Solutions**

```python
# Product reviews classification
reviews = [
    "This phone case is okay, nothing special but it does protect my phone adequately.",
    "Absolutely love this case! Perfect fit, great protection, and looks amazing!",
    "Terrible quality. Broke after just one week of normal use. Complete waste of money.",
    "Good value for the price. Not premium quality but gets the job done.",
    "The case is beautiful but makes the phone very bulky. Mixed feelings about this purchase."
]

# 3-point classification solution
classification_prompt = """Classify each review as POSITIVE, NEGATIVE, or NEUTRAL and provide reasoning:

Reviews:
1. This phone case is okay, nothing special but it does protect my phone adequately.
2. Absolutely love this case! Perfect fit, great protection, and looks amazing!
3. Terrible quality. Broke after just one week of normal use. Complete waste of money.
4. Good value for the price. Not premium quality but gets the job done.
5. The case is beautiful but makes the phone very bulky. Mixed feelings about this purchase.
"""

sample_classification_solution = """
Review 1: NEUTRAL - Uses neutral language like "okay" and "adequately," shows satisfaction with basic function but no enthusiasm
Review 2: POSITIVE - Strong positive language: "absolutely love," "perfect," "amazing"
Review 3: NEGATIVE - Clear negative indicators: "terrible," "broke," "waste of money"
Review 4: POSITIVE - Despite mentioning limitations, overall tone is satisfied: "good value," "gets the job done"
Review 5: NEUTRAL - Mixed sentiment explicitly stated: acknowledges beauty but notes drawbacks
"""

# 5-star rating system solution
star_rating_prompt = """Rate each review on a 1-5 star scale and explain your reasoning:"""

sample_star_solution = """
Review 1: 3 stars - Functional but unremarkable, meets basic expectations
Review 2: 5 stars - Enthusiastic praise across multiple dimensions
Review 3: 1 star - Product failure and complete dissatisfaction
Review 4: 4 stars - Good value despite limitations, recommends with caveats
Review 5: 3 stars - Balanced pros and cons, mixed overall experience
"""

print("✅ Classification Keys: Look for emotional indicators, qualifier words, overall sentiment balance")
```


### **Exercise D2: Topic Classification - Sample Solutions**

```python
# News headlines classification
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

topic_prompt = """Classify each headline into one of these categories: Technology, Finance, Sports, Environment, Entertainment, Health, Business, Science

Headlines:
[list headlines]
"""

sample_topic_solution = """
1. "New AI breakthrough could revolutionize medical diagnosis" - TECHNOLOGY (AI focus) or HEALTH (medical application)
   *Note: Could be dual-classified, but AI breakthrough suggests Technology as primary*

2. "Stock market reaches record high as inflation concerns ease" - FINANCE (stock market, inflation)

3. "Local high school wins state championship in dramatic overtime" - SPORTS (championship, overtime)

4. "Climate change report shows urgent need for action" - ENVIRONMENT (climate change focus)

5. "Celebrity couple announces surprise wedding in private ceremony" - ENTERTAINMENT (celebrity news)

6. "New vaccine shows 95% effectiveness in clinical trials" - HEALTH (vaccine, clinical trials)

7. "Tech giant announces major layoffs affecting thousands" - BUSINESS (corporate layoffs) or TECHNOLOGY (tech company)
   *Note: Business impact suggests Business as primary category*

8. "Archaeological discovery sheds light on ancient civilization" - SCIENCE (archaeological research)
"""

print("✅ Classification Strategy: Identify key domain-specific terms, consider primary vs secondary categories")
```


***

## **🗂️ Tables/JSON Generation Solutions**

### **Exercise E1: Data Structure Creation - Sample Solutions**

```python
# Recipe JSON solution
recipe_prompt = "Create a JSON object representing a recipe with ingredients, instructions, prep time, and difficulty level"

sample_recipe_json = '''
{
  "recipe": {
    "title": "Classic Chocolate Chip Cookies",
    "description": "Soft and chewy homemade chocolate chip cookies",
    "prep_time_minutes": 15,
    "cook_time_minutes": 10,
    "total_time_minutes": 25,
    "difficulty_level": "Easy",
    "servings": 24,
    "ingredients": [
      {
        "item": "all-purpose flour",
        "amount": "2.25",
        "unit": "cups"
      },
      {
        "item": "baking soda",
        "amount": "1",
        "unit": "teaspoon"
      },
      {
        "item": "salt",
        "amount": "1",
        "unit": "teaspoon"
      },
      {
        "item": "butter, softened",
        "amount": "1",
        "unit": "cup"
      },
      {
        "item": "granulated sugar",
        "amount": "0.75",
        "unit": "cup"
      },
      {
        "item": "brown sugar, packed",
        "amount": "0.75",
        "unit": "cup"
      },
      {
        "item": "vanilla extract",
        "amount": "2",
        "unit": "teaspoons"
      },
      {
        "item": "large eggs",
        "amount": "2",
        "unit": "pieces"
      },
      {
        "item": "chocolate chips",
        "amount": "2",
        "unit": "cups"
      }
    ],
    "instructions": [
      {
        "step": 1,
        "description": "Preheat oven to 375°F (190°C)."
      },
      {
        "step": 2,
        "description": "In a medium bowl, whisk together flour, baking soda, and salt. Set aside."
      },
      {
        "step": 3,
        "description": "In a large bowl, cream together softened butter and both sugars until light and fluffy."
      },
      {
        "step": 4,
        "description": "Beat in eggs one at a time, then add vanilla extract."
      },
      {
        "step": 5,
        "description": "Gradually mix in the flour mixture until just combined."
      },
      {
        "step": 6,
        "description": "Fold in chocolate chips."
      },
      {
        "step": 7,
        "description": "Drop rounded tablespoons of dough onto ungreased baking sheets."
      },
      {
        "step": 8,
        "description": "Bake for 9-11 minutes or until golden brown around edges."
      },
      {
        "step": 9,
        "description": "Cool on baking sheet for 2 minutes before transferring to wire rack."
      }
    ],
    "nutritional_info": {
      "calories_per_serving": 180,
      "fat_grams": 8,
      "carbs_grams": 26,
      "protein_grams": 2,
      "sugar_grams": 16
    },
    "tags": ["dessert", "baking", "cookies", "chocolate", "comfort food"]
  }
}
'''

# Phone comparison table solution
phone_table_prompt = "Generate a table comparing iPhone, Samsung Galaxy, and Google Pixel phones across price, camera, battery, and storage"

sample_phone_table = '''
| Feature | iPhone 15 Pro | Samsung Galaxy S24 | Google Pixel 8 Pro |
|---------|---------------|-------------------|-------------------|
| **Price** | $999-$1,199 | $799-$919 | $999-$1,059 |
| **Display** | 6.1" Super Retina XDR | 6.2" Dynamic AMOLED 2X | 6.7" LTPO OLED |
| **Camera** | 48MP main, 12MP ultra-wide, 12MP telephoto | 50MP main, 12MP ultra-wide, 10MP telephoto | 50MP main, 48MP ultra-wide, 48MP telephoto |
| **Battery** | Up to 23 hours video | 4,000 mAh (up to 22 hours) | 5,050 mAh (up to 24 hours) |
| **Storage** | 128GB, 256GB, 512GB, 1TB | 128GB, 256GB, 512GB | 128GB, 256GB, 512GB, 1TB |
| **Processor** | A17 Pro | Snapdragon 8 Gen 3 | Google Tensor G3 |
| **OS** | iOS 17 | Android 14 with One UI | Android 14 (stock) |
| **Build** | Titanium frame | Aluminum frame | Aluminum frame |
| **Water Resistance** | IP68 | IP68 | IP68 |
| **Special Features** | Action Button, USB-C | S Pen support, DEX mode | Magic Eraser, Call Screen |
'''

print("✅ JSON Keys: Proper nesting, consistent data types, comprehensive structure")
print("✅ Table Keys: Clear headers, consistent formatting, relevant comparisons")
```


### **Exercise E2: Complex Data Structures - Sample Solutions**

```python
# University course catalog solution
complex_course_prompt = """Create a JSON object for a university course catalog entry including:
- Course information (code, title, credits, description)
- Prerequisites (array of course codes)  
- Schedule options (multiple sections with days, times, instructor)
- Assessment methods (exams, assignments with weights)"""

sample_course_json = '''
{
  "course": {
    "course_code": "CS-301",
    "title": "Data Structures and Algorithms", 
    "department": "Computer Science",
    "credits": 4,
    "description": "Comprehensive study of fundamental data structures including arrays, linked lists, stacks, queues, trees, and graphs. Analysis of algorithm complexity and implementation of sorting and searching algorithms.",
    "learning_objectives": [
      "Implement and analyze fundamental data structures",
      "Design efficient algorithms for common problems", 
      "Evaluate time and space complexity using Big O notation",
      "Apply appropriate data structures to solve real-world problems"
    ],
    "prerequisites": [
      {
        "course_code": "CS-201",
        "title": "Introduction to Programming",
        "required": true
      },
      {
        "course_code": "MATH-151", 
        "title": "Discrete Mathematics",
        "required": true
      },
      {
        "course_code": "CS-250",
        "title": "Computer Organization",
        "required": false,
        "note": "Recommended but not required"
      }
    ],
    "sections": [
      {
        "section_number": "001",
        "instructor": {
          "name": "Dr. Sarah Johnson",
          "email": "s.johnson@university.edu",
          "office": "Engineering Building 312",
          "office_hours": "MWF 2:00-3:00 PM"
        },
        "schedule": {
          "days": ["Monday", "Wednesday", "Friday"],
          "time": "10:00 AM - 10:50 AM",
          "location": "ENGR 150",
          "capacity": 35,
          "enrolled": 28
        },
        "lab_section": {
          "days": ["Tuesday"],
          "time": "2:00 PM - 4:50 PM", 
          "location": "COMP LAB 201",
          "instructor": "Teaching Assistant"
        }
      },
      {
        "section_number": "002",
        "instructor": {
          "name": "Prof. Michael Chen",
          "email": "m.chen@university.edu",
          "office": "Engineering Building 315",
          "office_hours": "TTh 1:00-2:30 PM"
        },
        "schedule": {
          "days": ["Tuesday", "Thursday"],
          "time": "11:00 AM - 12:15 PM",
          "location": "ENGR 200",
          "capacity": 40,
          "enrolled": 35
        },
        "lab_section": {
          "days": ["Wednesday"],
          "time": "3:00 PM - 5:50 PM",
          "location": "COMP LAB 203", 
          "instructor": "Teaching Assistant"
        }
      }
    ],
    "assessment_methods": [
      {
        "type": "Midterm Exams",
        "count": 2,
        "weight_percentage": 25,
        "description": "Two in-class examinations covering theoretical concepts"
      },
      {
        "type": "Final Exam",
        "count": 1,
        "weight_percentage": 25,
        "description": "Comprehensive final examination"
      },
      {
        "type": "Programming Assignments",
        "count": 6,
        "weight_percentage": 35,
        "description": "Implementation projects using various data structures"
      },
      {
        "type": "Lab Exercises",
        "count": 10,
        "weight_percentage": 10,
        "description": "Weekly hands-on programming practice"
      },
      {
        "type": "Participation",
        "count": 1,
        "weight_percentage": 5,
        "description": "Class attendance and engagement"
      }
    ],
    "required_materials": [
      {
        "type": "textbook",
        "title": "Introduction to Algorithms",
        "authors": ["Thomas H. Cormen", "Charles E. Leiserson"],
        "isbn": "978-0262033848",
        "required": true
      },
      {
        "type": "software",
        "name": "Python 3.9+",
        "cost": "Free",
        "required": true
      }
    ],
    "grading_scale": {
      "A": "90-100%",
      "B": "80-89%", 
      "C": "70-79%",
      "D": "60-69%",
      "F": "Below 60%"
    }
  }
}
'''

print("✅ Complex Structure Keys: Nested objects, arrays of objects, comprehensive details, real-world applicability")
```


***

## **⚖️ Evaluation/Comparison Solutions**

### **Exercise F1: Algorithm Comparison - Sample Solutions**

```python
# Sorting algorithms comparison
comparison_prompt = """Compare bubble sort, quicksort, and merge sort across:
- Time complexity (best, average, worst case)
- Space complexity  
- Stability
- Best use cases
- Implementation difficulty"""

sample_algorithm_comparison = '''
# Sorting Algorithms Comparison

## Time Complexity Analysis
| Algorithm | Best Case | Average Case | Worst Case |
|-----------|-----------|--------------|------------|
| Bubble Sort | O(n) | O(n²) | O(n²) |
| Quicksort | O(n log n) | O(n log n) | O(n²) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) |

## Space Complexity
- **Bubble Sort**: O(1) - In-place sorting
- **Quicksort**: O(log n) - Recursive stack space
- **Merge Sort**: O(n) - Additional array needed for merging

## Stability
- **Bubble Sort**: ✅ Stable - Equal elements maintain relative order
- **Quicksort**: ❌ Unstable - Partitioning can change relative order
- **Merge Sort**: ✅ Stable - Merge process preserves order

## Best Use Cases
- **Bubble Sort**: 
  - Educational purposes only
  - Very small datasets (< 10 elements)
  - Nearly sorted data
  
- **Quicksort**:
  - General-purpose sorting
  - Large datasets with random distribution
  - When space is limited
  - Built-in library implementations

- **Merge Sort**:
  - When stability is required
  - Guaranteed O(n log n) performance needed
  - External sorting (large datasets that don't fit in memory)
  - Linked list sorting

## Implementation Difficulty
1. **Bubble Sort**: Easiest - Simple nested loops
2. **Quicksort**: Moderate - Requires understanding of partitioning
3. **Merge Sort**: Moderate - Requires understanding of divide-and-conquer

## Practical Recommendations
- **For learning**: Start with bubble sort, then merge sort
- **For production**: Use quicksort (or hybrid algorithms like Introsort)
- **For stability requirements**: Use merge sort
- **For embedded systems**: Consider quicksort for space efficiency
'''

print("✅ Comparison Framework: Multiple dimensions, clear criteria, practical recommendations")
```


### **Exercise F2: Product Evaluation - Sample Solutions**

```python
# Streaming services comparison
streaming_prompt = "Compare Netflix, Disney+, and HBO Max for a family with young children and teenagers"

sample_streaming_comparison = '''
# Streaming Service Comparison for Families

## Content Library Analysis

### Netflix
**Strengths:**
- Largest overall content library (15,000+ titles)
- Strong original content for all ages
- Excellent international content variety
- Robust kids' section with parental controls
- Teen-focused originals (Stranger Things, Riverdale)

**Weaknesses:**
- Content rotates frequently
- Some mature content mixed with family content
- Higher cost for premium features

### Disney+
**Strengths:**
- Best for young children (Disney, Pixar, Marvel)
- Family-friendly guarantee - all content appropriate
- High-quality productions and classics
- Educational content (National Geographic)
- Affordable pricing

**Weaknesses:**
- Limited content for teenagers and adults
- Smaller overall library
- Less frequent new releases

### HBO Max
**Strengths:**
- Premium quality content and originals
- Strong teen/young adult programming
- Warner Bros movie releases
- Cartoon Network content for kids

**Weaknesses:**
- Most expensive option
- Heavy on mature content
- Fewer options for very young children

## Family Suitability Scores (1-10)

| Criteria | Netflix | Disney+ | HBO Max |
|----------|---------|---------|---------|
| Young Children (5-10) | 8/10 | 10/10 | 6/10 |
| Teenagers (13-17) | 9/10 | 5/10 | 8/10 |
| Family Movie Nights | 8/10 | 10/10 | 7/10 |
| Educational Value | 7/10 | 9/10 | 6/10 |
| Content Safety | 7/10 | 10/10 | 5/10 |
| Value for Money | 7/10 | 9/10 | 6/10 |

## Recommendation
**For your family situation**: Start with **Disney+** as the primary service for safe, family-friendly content that works for all ages. Add **Netflix** as a secondary service to satisfy teenagers' interests and provide more variety for parents.

**Budget-conscious option**: Disney+ only initially, add Netflix later
**Best experience**: Disney+ + Netflix combination
**Premium option**: All three services for maximum choice
'''

print("✅ Product Evaluation Keys: User-specific criteria, quantified comparisons, clear recommendations")
```


### **Exercise F3: Writing Quality Assessment - Sample Solutions**

```python
# Email versions comparison
email_comparison_prompt = """Evaluate these three email versions for professionalism, tone, clarity, and effectiveness:

Email 1: [brief version]
Email 2: [overly formal version]  
Email 3: [balanced version]"""

sample_email_evaluation = '''
# Email Communication Analysis

## Individual Assessment

### Email Version 1: "Hi, I need the report by tomorrow. Please send it ASAP. Thanks."

**Strengths:**
- Clear and direct request
- Brief and to the point
- Easy to understand

**Weaknesses:**
- Too casual for professional setting
- Demanding tone ("I need", "ASAP")
- No context or explanation
- Lacks courtesy and consideration
- No acknowledgment of inconvenience

**Overall Score: 4/10** - Gets the message across but poor professionalism

### Email Version 2: [Long, overly formal version]

**Strengths:**
- Very polite and respectful
- Acknowledges the inconvenience
- Provides context for urgency
- Shows flexibility

**Weaknesses:**
- Excessively wordy and verbose
- Takes too long to get to the point
- May seem insincere due to over-formality
- Could lose reader's attention

**Overall Score: 6/10** - Professional but inefficient communication

### Email Version 3: [Balanced version]

**Strengths:**
- Professional yet friendly tone
- Clear request with context
- Acknowledges potential difficulty
- Shows appreciation and flexibility
- Appropriate length and structure
- Maintains professional relationships

**Weaknesses:**
- Minor: Could be slightly more specific about deadline time

**Overall Score: 9/10** - Excellent balance of professionalism and efficiency

## Comparative Analysis

| Criteria | Email 1 | Email 2 | Email 3 |
|----------|---------|---------|---------|
| **Clarity** | High | Medium | High |
| **Professionalism** | Low | High | High |
| **Efficiency** | High | Low | High |
| **Tone** | Demanding | Overly formal | Balanced |
| **Relationship Building** | Poor | Good | Excellent |
| **Likelihood of Success** | Medium | Good | High |

## Key Learning Points

1. **Balance is crucial**: Neither too brief nor too verbose
2. **Context matters**: Explain urgency without being demanding
3. **Show consideration**: Acknowledge the recipient's situation
4. **Maintain relationships**: Professional communication builds long-term cooperation
5. **Call to action**: Be clear about what you need and when

## Best Practices Demonstrated
- Start with appropriate greeting
- Provide context for requests
- Acknowledge potential inconvenience
- Use collaborative language ("would it be possible")
- Express genuine appreciation
- Offer to help if needed
- End with professional closing
'''

print("✅ Writing Assessment Keys: Multiple criteria evaluation, specific examples, actionable feedback")
```


***

## **🎯 Challenge Exercise Solutions**

### **Challenge 1: Multi-Modal Content Creation - Sample Solution**

```python
# Binary search mini-lesson solution
binary_search_lesson_prompt = """Create a mini-lesson on binary search that includes:
1. A simple explanation for beginners
2. Python code implementation with comments  
3. JSON data showing step-by-step example
4. Comparison table: Binary vs Linear search
5. Classification of when to use each method"""

sample_binary_search_lesson = '''
# Binary Search Mini-Lesson

## 1. Simple Explanation
Binary search is like looking up a word in a dictionary. Instead of checking every page from the beginning, you open to the middle and see if your word comes before or after that page. Then you eliminate half the dictionary and repeat the process. This makes finding things much faster than checking every item one by one.

**Key Requirements**: The list must be sorted first!

## 2. Python Implementation
```

def binary_search(arr, target):
"""
Search for target in a sorted array using binary search.

    Args:
        arr: Sorted list of elements
        target: Element to search for
        
    Returns:
        Index of target if found, -1 if not found
    """
    left = 0                    # Start of search range
    right = len(arr) - 1        # End of search range
    
    while left <= right:
        # Find middle index (avoid overflow)
        mid = left + (right - left) // 2
        
        # Check if target is at middle
        if arr[mid] == target:
            return mid
        
        # If target is smaller, search left half
        elif arr[mid] > target:
            right = mid - 1
        
        # If target is larger, search right half
        else:
            left = mid + 1
    
    # Target not found
    return -1
    
# Example usage

numbers =
result = binary_search(numbers, 7)
print(f"Found 7 at index: {result}")

```

## 3. Step-by-Step Example (JSON)
```

{
"binary_search_example": {
"array": ,
"target": 7,
"steps": [
{
"step": 1,
"left": 0,
"right": 7,
"mid": 3,
"mid_value": 7,
"comparison": "arr == 7",
"action": "Found! Return index 3",
"remaining_elements": 8
}
],
"result": {
"found": true,
"index": 3,
"comparisons_made": 1,
"efficiency": "Best case - found on first try"
}
},
"worst_case_example": {
"array": ,
"target": 1,
"steps": [
{
"step": 1,
"left": 0,
"right": 7,
"mid": 3,
"mid_value": 7,
"comparison": "arr > 1",
"action": "Search left half",
"remaining_elements": 4
},
{
"step": 2,
"left": 0,
"right": 2,
"mid": 1,
"mid_value": 3,
"comparison": "arr > 1",
"action": "Search left half",
"remaining_elements": 2
},
{
"step": 3,
"left": 0,
"right": 0,
"mid": 0,
"mid_value": 1,
"comparison": "arr == 1",
"action": "Found! Return index 0",
"remaining_elements": 1
}
],
"result": {
"found": true,
"index": 0,
"comparisons_made": 3,
"efficiency": "Log₂(8) = 3 comparisons maximum"
}
}
}

```

## 4. Comparison Table

| Feature | Binary Search | Linear Search |
|---------|---------------|---------------|
| **Time Complexity** | O(log n) | O(n) |
| **Best Case** | O(1) | O(1) |
| **Worst Case** | O(log n) | O(n) |
| **Average Case** | O(log n) | O(n/2) |
| **Space Complexity** | O(1) iterative, O(log n) recursive | O(1) |
| **Prerequisites** | Array must be sorted | No requirements |
| **Implementation** | More complex | Simple |
| **When to Use** | Large sorted datasets | Small or unsorted data |

## 5. Usage Classification

### Use Binary Search When:
- **Large datasets** (1000+ elements)
- **Data is already sorted** or sorting cost is justified
- **Frequent searches** on the same dataset
- **Performance is critical**
- **Memory usage should be minimal**

### Use Linear Search When:
- **Small datasets** (< 100 elements)
- **Unsorted data** and sorting isn't worth the cost
- **Infrequent searches**
- **Simple implementation needed**
- **Data changes frequently** (insertions/deletions)

### Performance Comparison Example:
- **1,000 elements**: Binary = 10 comparisons max, Linear = 500 average
- **1,000,000 elements**: Binary = 20 comparisons max, Linear = 500,000 average
- **1 billion elements**: Binary = 30 comparisons max, Linear = 500 million average

**Key Insight**: Binary search's advantage grows exponentially with data size!
'''

print("✅ Multi-Modal Success: Combines explanation, code, data, comparison, and classification effectively")
```


***

## **📈 Assessment Rubric with Solutions**

### **Prompt Quality Indicators**

```python
# Examples of good vs poor prompts:

# POOR PROMPT
poor_example = "Write code"

# GOOD PROMPT  
good_example = """Write a Python function that validates email addresses using regular expressions. 
Include:
- Function docstring with parameter and return descriptions
- Input validation for None/empty strings
- At least 3 test cases showing valid and invalid emails
- Comments explaining the regex pattern"""

# EXCELLENT PROMPT
excellent_example = """Create a comprehensive email validation system in Python that includes:

1. A main validation function using regex that checks for:
   - Valid characters before and after @
   - Proper domain format
   - Common TLD validation

2. Edge case handling:
   - None/empty inputs
   - Whitespace trimming
   - Case insensitivity

3. A test suite with:
   - 5 valid email examples
   - 5 invalid email examples  
   - Expected results for each

4. Documentation:
   - Detailed docstrings
   - Inline comments explaining regex components
   - Usage examples

Format the response with clear sections and proper code formatting."""

print("✅ Prompt Quality Progression: Specific requirements, clear structure, comprehensive scope")
```


### **Solution Quality Framework**

```python
# Framework for evaluating GPT-4o-mini responses:

quality_framework = '''
## Response Quality Assessment (1-5 scale)

### Technical Accuracy (Weight: 30%)
5: Code runs perfectly, logic is correct, handles edge cases
4: Code works with minor issues, mostly correct logic
3: Code works for basic cases, some logical gaps
2: Code has errors but shows understanding
1: Code doesn't work, major conceptual errors

### Completeness (Weight: 25%)
5: Addresses all prompt requirements thoroughly  
4: Addresses most requirements with good detail
3: Covers main points but lacks some detail
2: Partial coverage of requirements
1: Minimal coverage, misses key elements

### Code Quality (Weight: 20%)
5: Clean, readable, well-documented, follows best practices
4: Good structure with adequate documentation
3: Functional code with basic documentation
2: Messy but working code
1: Poor structure, no documentation

### Practical Utility (Weight: 15%)
5: Ready for production use, highly practical
4: Needs minor tweaks for real-world use  
3: Good starting point, needs development
2: Demonstrates concept but not practical
1: Academic exercise only

### Innovation/Insight (Weight: 10%)
5: Shows creative problem-solving, unique approaches
4: Some creative elements or insights
3: Standard approach, competently executed
2: Basic solution, no special insights
1: Minimal effort or understanding

## Grade Calculation:
Total Score = (Technical × 0.3) + (Completeness × 0.25) + (Code Quality × 0.2) + (Utility × 0.15) + (Innovation × 0.1)

4.5-5.0: Excellent (A)
3.5-4.4: Good (B)  
2.5-3.4: Satisfactory (C)
1.5-2.4: Needs Improvement (D)
Below 1.5: Unsatisfactory (F)
'''

print("✅ Use this framework to evaluate your prompt engineering skills and GPT-4o-mini responses")
```


***

## **🎓 Final Practice Challenge**

```python
# Ultimate challenge combining all skills:
ultimate_challenge = """
CHALLENGE: Create a Personal Finance Assistant

Your task: Design prompts for GPT-4o-mini to create a comprehensive personal finance analysis system.

Required Components:
1. **Text Generation**: Write financial advice explanations for different age groups (20s, 40s, retirees)
2. **Code**: Create Python functions for budget calculations and investment projections  
3. **Summarization**: Summarize financial news articles and investment reports
4. **Classification**: Categorize expenses and classify financial goals by priority
5. **JSON/Tables**: Generate budget templates and investment portfolio structures
6. **Evaluation**: Compare different investment strategies and financial products

Success Criteria:
- All components work together as a cohesive system
- Code is production-ready with proper error handling
- Content is accurate and actionable
- Data structures are well-designed
- Comparisons are fair and comprehensive

Time Limit: 2 hours
Evaluation: Use the quality framework above

BONUS: Create a simple web interface design (HTML/CSS) for your finance assistant
"""

print("🏆 This challenge tests everything you've learned about prompt engineering!")
print("🎯 Focus on creating specific, detailed prompts that guide GPT-4o-mini to produce high-quality results")
```


***

**💡 Key Success Factors from Solutions:**

1. **Specificity**: Detailed prompts produce better results
2. **Structure**: Organize requests clearly with numbered requirements
3. **Context**: Provide background and use cases
4. **Examples**: Show desired format/style when possible
5. **Constraints**: Set clear parameters (length, format, audience)
6. **Iteration**: Refine prompts based on initial results
7. **Validation**: Always test code and verify data accuracy

**🚀 Ready to become a prompt engineering expert? Practice these solutions and create your own variations!**

