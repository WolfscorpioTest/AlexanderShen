let valueDisplays = document.querySelectorAll(".num");
let interval = 1500;
var delayInMilliseconds = 1300; //1 second

const header = document.querySelector("stats-wrapper");
const sectionOne = document.querySelector(".stats-wrapper");

const sectionOneOptions = {
    rootMargin: "-200px 0px 0px 0px"
};

const sectionOneObserver = new IntersectionObserver(function (
    entries,
    sectionOneObserver
) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            setTimeout(function () {
                valueDisplays.forEach((valueDisplay) => {
                    let startValue = 0;
                    let endValue = parseInt(valueDisplay.getAttribute("data-val"));
                    let duration = Math.floor(interval / endValue);
                    let counter = setInterval(function () {

                        if (endValue == 4) {
                            startValue += 1;
                        }
                        else {
                            startValue += 4;
                        }
                        valueDisplay.textContent = startValue;

                        if (startValue == endValue) {
                            clearInterval(counter);
                        }
                    }, duration);
                });
            }, delayInMilliseconds);
        }

    });
},
    sectionOneOptions);

sectionOneObserver.observe(sectionOne);

const appearOnScroll = new IntersectionObserver(function(
    entries,
    appearOnScroll
  ) {
    entries.forEach(entry => {
      if (!entry.isIntersecting) {
        return;
      } else {
        entry.target.classList.add("appear");
        appearOnScroll.unobserve(entry.target);
      }
    });
  },
  appearOptions);

