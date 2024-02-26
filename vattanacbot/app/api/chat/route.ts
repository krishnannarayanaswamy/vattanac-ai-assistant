import OpenAI from 'openai';
import {OpenAIStream, StreamingTextResponse} from 'ai';
import {AstraDB} from "@datastax/astra-db-ts";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const astraDb = new AstraDB(process.env.ASTRA_DB_APPLICATION_TOKEN, process.env.ASTRA_DB_ID, process.env.ASTRA_DB_REGION, process.env.ASTRA_DB_NAMESPACE);

export async function POST(req: Request) {
  try {
    const {messages, useRag, llm, similarityMetric} = await req.json();

    const latestMessage = messages[messages?.length - 1]?.content;

    let docContext = '';
    if (useRag) {
      
      const translatePrompt = [
        {
          role: 'system',
          content: `Translate to English, if the input is not english. If the input is English, just return the english query back unaltered".
          `,
        },
      ]
      
      const completiondata = await openai.chat.completions.create(
        {
          model: 'gpt-4-0125-preview',
          messages: [...translatePrompt, ...messages],
        }
      );

      const datatranslated = completiondata.choices[0]?.message?.content;

      console.log(datatranslated);

      const {data} = await openai.embeddings.create({input: datatranslated, model: 'text-embedding-ada-002'});

      const collection = await astraDb.collection(`vattanac_chatbot`);

      const cursor= collection.find(null, {
        sort: {
          $vector: data[0]?.embedding,
        },
        limit: 5,
      });
      
      const documents = await cursor.toArray();
      
      docContext = `
        START CONTEXT
        ${documents?.map(doc => doc.content).join("\n")}
        END CONTEXT
      `
    }

    console.log(docContext);

    const ragPrompt = [
      {
        role: 'system',
        content: `You are an AI assistant assisting customers to about Vattanac Bank in Cambodia. Include the product description when responding with the list of product recommendation. Answer question based on the context information which is extracted from their webpage. Format responses using markdown where applicable. All the responses should be the same language as the user used. Respond in English, if user query was in english. Convert the context data to Khmer, if user query is in Khmer.
        ${docContext} 
        If the answer is not provided in the context, the AI assistant will say, "I'm sorry, I don't know the answer".
        `,
      },
    ]


    const response = await openai.chat.completions.create(
      {
        model: 'gpt-4-0125-preview',
        stream: true,
        messages: [...ragPrompt, ...messages],
      }
    );
    const stream = OpenAIStream(response);
    return new StreamingTextResponse(stream);
  } catch (e) {
    throw e;
  }
}
