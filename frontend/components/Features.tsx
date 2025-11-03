import { AnimatedTestimonials } from "./ui/animated-testimonials";

function AnimatedTestimonialsDemo() {
  const testimonials = [
    {
      quote:
        "Honestly, this is the first time I've been able to run powerful models without worrying about sending sensitive data to third parties. The TEE sharding just works.",
      name: "Peggy Chen",
      designation: "Product Manager @ Atlassian",
      src: "/images/girl.png",
    },
    {
      quote:
        "TeeTee is a game-changer. The fact that our data stays private while the model is split like a pizza is kind of awesome. Everyone gets a slice, nobody gets your toppings.",
      name: "Michael Rodriguez",
      designation: "Senior Developer @ NewRelic",
      src: "/images/angmo23.png",
    },
    {
      quote:
        "I love the cost-sharing concept that purpose by TeeTee. Instead of over-provisioning GPUs that sit idle, we get top-tier performance without draining our budget.",
      name: "Emily Clare",
      designation: "Operations Director @ Databricks",
      src: "/images/lady4.png",
    },
    {
      quote:
        "No joke, distributed inference was the missing piece. We get the performance we need without dropping six figures on infrastructure. It's pretty clever engineering.",
      name: "James Smith",
      designation: "Engineering Lead @ DataDog",
      src: "/images/blue.png",
    },
    {
      quote:
        "The setup was smoother than expected. The distributed inference felt almost seamless, even though we knew the model was running on multiple secure nodes.",
      name: "Benjamin Thompson",
      designation: "Software Engineer @ kubernetes",
      src: "/images/angmo4.png",
    },
  ];
  return (
    <section className="bg-gradient-to-r from-violet-100/40 via-purple-100/30 to-violet-100/40">
      <AnimatedTestimonials testimonials={testimonials} />
    </section>
  );
}

export { AnimatedTestimonialsDemo };
