import React, { useEffect, useRef, useState } from "react";

export const CallToAction = () => {
    const [openIndex, setOpenIndex] = React.useState<number | null>(null);
    const [isVisible, setIsVisible] = useState(false);
    const sectionRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                setIsVisible(entry.isIntersecting);
            },
            {
                threshold: 0.2,
                rootMargin: '0px'
            }
        );

        const currentRef = sectionRef.current;
        if (currentRef) {
            observer.observe(currentRef);
        }

        return () => {
            if (currentRef) {
                observer.unobserve(currentRef);
            }
        };
    }, []);

    const faqs = [
        {
            question: "How does TeeTee allow organizations to run large LLMs without needing massive infrastructure?",
            answer: "TeeTee uses a distributed sharding mechanism that splits a large LLM into smaller components, with each shard hosted inside a secure Trusted Execution Environment (TEE). Instead of requiring a single organization to run the entire model, multiple participants host different shards. Together, they enable full-scale inference while each pays only a portion of the overall infrastructure cost.",
        },
        {
            question: "How does TeeTee keep my data private during inference?",
            answer: "All computation takes place within TEEs, ensuring data and model parameters remain encrypted and inaccessible to the host machines. Because no single party holds the full model and all communication is encrypted end-to-end, TeeTee prevents data exposure—even from other shard participants or infrastructure providers.",
        },
        {
            question: "Will distribution across TEEs slow down model performance?",
            answer: "Surprisingly little. TeeTee is architected for efficient cross-shard execution, minimizing latency while maintaining high model throughput. Because each TEE processes only a portion of the workload, performance remains comparable to centralized deployments—despite the added security and decentralization benefits.",
        },
        {
            question: "What do I gain by contributing my infrastructure to host a shard?",
            answer: "By hosting a shard, organizations gain affordable access to a powerful LLM they otherwise couldn't afford on their own. In addition, participants can earn revenue when external users call the model through APIs—creating economic incentive while still benefiting from enterprise-grade privacy and security.",
        },
    ];
    return (
        <>
            <style>{`
                @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
            
                * {
                    font-family: 'Poppins', sans-serif;
                }

                .faq-section {
                    opacity: 0;
                    transform: translateY(50px);
                    transition: opacity 0.8s ease-out, transform 0.8s ease-out;
                }

                .faq-section.visible {
                    opacity: 1;
                    transform: translateY(0);
                }

                @keyframes jump {
                    0% {
                        transform: translateY(0);
                    }
                    12% {
                        transform: translateY(-15px);
                    }
                    24% {
                        transform: translateY(0);
                    }
                    24%, 100% {
                        transform: translateY(0);
                    }
                }

                .jump-animation {
                    display: inline-block;
                    animation: jump 2.5s ease-in-out infinite;
                }
            `}</style>
            <div ref={sectionRef} className={`max-w-5xl mx-auto flex flex-col items-center justify-center px-4 py-20 faq-section ${isVisible ? 'visible' : ''}`}>
                <div className="w-full">
                    <p className="text-indigo-600 text-lg font-medium text-center">FAQ's</p>
                    <h1 className="text-6xl font-semibold text-center mt-4">
                        Looking for answer <span className="jump-animation">?</span>
                    </h1>
                    <p className="text-3xl text-slate-500 mt-6 pb-12 text-center">
                        Everything you need to know about TeeTee's distributed AI infrastructure and how it keeps your data secure.
                    </p>
                    {faqs.map((faq, index) => (
                        <div className="border-b border-slate-200 py-8 cursor-pointer" key={index} onClick={() => setOpenIndex(openIndex === index ? null : index)}>
                            <div className="flex items-center justify-between gap-4">
                                <h3 className="text-3xl font-medium">
                                    {faq.question}
                                </h3>
                                <svg width="32" height="32" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg" className={`${openIndex === index ? "rotate-180" : ""} transition-all duration-500 ease-in-out flex-shrink-0`}>
                                    <path d="m4.5 7.2 3.793 3.793a1 1 0 0 0 1.414 0L13.5 7.2" stroke="#1D293D" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                            </div>
                            <p className={`text-xl text-slate-500 transition-all duration-500 ease-in-out ${openIndex === index ? "opacity-100 max-h-[500px] translate-y-0 pt-6" : "opacity-0 max-h-0 -translate-y-2"}`} >
                                {faq.answer}
                            </p>
                        </div>
                    ))}
                </div>
            </div>
        </>
    );
};
