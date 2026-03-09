from openai import OpenAI

a4f_api_key = "ddc-a4f-46ad914fc8644db38e31a214de48bc74"
a4f_base_url = "https://api.a4f.co/v1"


client = OpenAI(
    api_key=a4f_api_key,
    base_url=a4f_base_url,
)
while True:
  prompt = str(input('You : '))
  completion = client.chat.completions.create(
    model="provider-5/gpt-4o-mini",
    messages=[
      {"role": "user", "content": prompt}
    ]
  )

  AI = completion.choices[0].message.content
  print(f'AI: {AI}')