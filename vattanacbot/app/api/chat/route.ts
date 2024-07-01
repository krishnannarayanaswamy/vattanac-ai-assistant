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
          content: `The input might be in Khmer language. Translate to English, if the input is not english. If the input is English, just return the english query back unaltered".
          `,
        },
      ]
      
      const completiondata = await openai.chat.completions.create(
        {
          model: 'gpt-4o',
          messages: [...translatePrompt, ...messages],
        }
      );

      const datatranslated = completiondata.choices[0]?.message?.content;

      //const datatranslated = latestMessage;

      console.log(datatranslated);

      const {data} = await openai.embeddings.create({input: datatranslated, model: 'text-embedding-3-large', dimensions: 1024});

      const collection = await astraDb.collection(`vattanac_bank_year`);

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
        content: `You are an AI Assistant for Vattanac Bank in cambodia, tasked with answering any question about finanical documents such as annual reports. Format responses using markdown where applicable. The context data might contain data from tables and graphs.The context data is in english language.  Respond in English, if user query was in english. Respond in Khmer, if the user query is in Khmer.
        ${docContext} 
        If the answer is not provided in the context, the AI assistant will say, "I'm sorry, I don't know the answer". Remember to respond in the same language as the user asked the question.
        `,
      },
    ]


    const response = await openai.chat.completions.create(
      {
        model: 'gpt-4o',
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
